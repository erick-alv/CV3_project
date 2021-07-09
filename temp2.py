using_local = True
if using_local:
    root_dir = "cv3dst_exercise"
    gnn_root_dir = "cv3dst_gnn_exercise"
import os
import sys
sys.path.append(os.path.join(gnn_root_dir, 'src'))

from torch.nn import functional as F
import numpy as np
import time


import torch
from torch import nn

from tracker.data_track import MOT16Sequences
from tracker.tracker import Tracker, ReIDTracker
from tracker.utils import run_tracker, cosine_distance
from scipy.optimize import linear_sum_assignment as linear_assignment
import os.path as osp

import motmetrics as mm
mm.lap.default_solver = 'lap'


_UNMATCHED_COST = 255


class ReIDHungarianTracker(ReIDTracker):
    def data_association(self, boxes, scores, pred_features):
        """Refactored from previous implementation to split it onto distance computation and track management"""
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

            distance = self.compute_distance_matrix(track_features, pred_features,
                                                    track_boxes, boxes, metric_fn=cosine_distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)


        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        """Updates existing tracks and removes unmatched tracks.
           Reminder: If the costs are equal to _UNMATCHED_COST, it's not a
           match.
        """
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)
            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        self.tracks = [t for t in self.tracks
                       if t.id not in unmatched_track_ids]

        # Add new tracks.
        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


class LongTermReIDHungarianTracker(ReIDHungarianTracker):
    def __init__(self, patience, *args, **kwargs):
        """ Add a patience parameter"""
        self.patience = patience
        super().__init__(*args, **kwargs)

    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0:  # Only change
                self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)

            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])

                # Note: the track is matched, therefore, inactive is set to 0
                self.tracks[track_idx].inactive = 0
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        ##################
        ### TODO starts
        ##################

        # Update the `inactive` attribute for those tracks that have been
        # not been matched. kill those for which the inactive parameter
        # is > self.patience
        for i in range(len(self.tracks)):
            if self.tracks[i].id in unmatched_track_ids:
                self.tracks[i].inactive += 1

        self.tracks = [t for t in self.tracks if t.inactive <= self.patience]  # <-- Needs to be updated

        ##################
        ### TODO ends
        ##################

        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.):
        super().__init__()

        edge_in_dim = 2 * node_dim + 2 * edge_dim  # 2*edge_dim since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(*[nn.Linear(edge_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout)])

        node_in_dim = node_dim + edge_dim
        self.node_mlp = nn.Sequential(*[nn.Linear(node_in_dim, node_dim), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Dropout(dropout)])

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_a_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim)
        """

        n_nodes_a, n_nodes_b, _ = edge_embeds.shape

        ########################
        #### TODO starts
        ########################

        a_repl = torch.stack([nodes_a_embeds] * n_nodes_b, dim=1).cuda()
        b_repl = torch.stack([nodes_b_embeds] * n_nodes_a, dim=0).cuda()
        edge_in = torch.cat([edge_embeds, a_repl, b_repl],
                            dim=2).cuda()  # has shape (|A|, |B|, 2*node_dim + 2*edge_dim)

        ########################
        #### TODO ends
        ########################

        return self.edge_mlp(edge_in)

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim )
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            tuple(
                updated_nodes_a_embeds: torch.Tensor with shape (|A|, node_dim),
                updated_nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
                )
        """

        ########################
        #### TODO starts
        ########################

        # NOTE: Use 'sum' as aggregation function
        sum_for_a = torch.sum(edge_embeds, dim=1)
        sum_for_b = torch.sum(edge_embeds, dim=0)

        nodes_a_in = torch.cat([nodes_a_embeds, sum_for_a], dim=1)  # Has shape (|A|, node_dim + edge_dim)
        nodes_b_in = torch.cat([nodes_b_embeds, sum_for_b], dim=1)  # Has shape (|B|, node_dim + edge_dim)

        ########################
        #### TODO ends
        ########################

        nodes_a = self.node_mlp(nodes_a_in)
        nodes_b = self.node_mlp(nodes_b_in)

        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(edge_embeds, nodes_a_embeds, nodes_b_embeds)
        nodes_a_latent, nodes_b_latent = self.node_update(edge_embeds_latent, nodes_a_embeds, nodes_b_embeds)

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent


class AssignmentSimilarityNet(nn.Module):
    def __init__(self, reid_network, node_dim, edge_dim, reid_dim, edges_in_dim, num_steps, dropout=0.):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(
            *[nn.Linear(edges_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(edge_dim, edge_dim),
              nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = nn.Sequential(*[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)])

    def compute_edge_feats(self, track_coords, current_coords, track_t, curr_t):
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)


        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features
        """

        ########################
        #### TODO starts
        ########################

        # NOTE 1: we recommend you to use box centers to compute distances
        # in the x and y coordinates.

        # NOTE 2: Check out the  code inside train_one_epoch function and
        # LongTrackTrainingDataset class a few cells below to debug this

        track_centers = track_coords.view(-1, 2, 2).mean(-2)
        curr_centers = current_coords.view(-1, 2, 2).mean(-2)

        track_w = track_coords[..., 2] - track_coords[..., 0]
        track_h = track_coords[..., 3] - track_coords[..., 1]

        curr_w = current_coords[..., 2] - current_coords[..., 0]
        curr_h = current_coords[..., 3] - current_coords[..., 1]

        assert (track_w > 0).all() and (track_h > 0).all()

        denom = (track_w.unsqueeze(1) + curr_w.unsqueeze(0)) / 2
        dist_x = (track_centers[..., 0].unsqueeze(1) - curr_centers[..., 0].unsqueeze(0))
        dist_y = (track_centers[..., 1].unsqueeze(1) - curr_centers[..., 1].unsqueeze(0))

        dist_x = dist_x / denom
        dist_y = dist_y / denom

        dist_w = torch.log(track_w.unsqueeze(1) / curr_w.unsqueeze(0))
        dist_h = torch.log(track_h.unsqueeze(1) / curr_h.unsqueeze(0))

        dist_t = (track_t.unsqueeze(1) - curr_t.unsqueeze(0)).type(dist_h.dtype).to(dist_h.device)

        edge_feats = torch.stack([dist_x, dist_y, dist_w, dist_h, dist_t])
        edge_feats = edge_feats.permute(1, 2, 0)

        ########################
        #### TODO ends
        ########################

        return edge_feats  # has shape (num_trakcs, num_boxes, 5)

    def forward(self, track_app, current_app, track_coords, current_coords, track_t, curr_t):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)

        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and
                             detection j are a match, according to the classifier at the given neural message passing step
        """

        # Get initial edge embeddings to
        dist_reid = cosine_distance(track_app, current_app)
        pos_edge_feats = self.compute_edge_feats(track_coords, current_coords, track_t, curr_t)
        edge_feats = torch.cat((pos_edge_feats, dist_reid.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        track_embeds = F.relu(self.cnn_linear(track_app))
        curr_embeds = F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)
            edge_embeds, track_embeds, curr_embeds = self.graph_net(edge_embeds=edge_embeds,
                                                                    nodes_a_embeds=track_embeds,
                                                                    nodes_b_embeds=curr_embeds)

            classified_edges.append(self.classifier(edge_embeds))

        return torch.stack(classified_edges).squeeze(-1)


class MPNTracker(LongTermReIDHungarianTracker):
    def __init__(self, assign_net, *args, **kwargs):
        self.assign_net = assign_net
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, pred_features):
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0).cuda()
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).cuda()
            track_t = torch.as_tensor([self.im_index - t.inactive - 1 for t in self.tracks]).cuda()

            ########################
            #### TODO starts
            ########################

            # Do a forward pass through self.assign_net to obtain our costs.
            # Note: self.assign_net will return unnormalized probabilities.
            # Make sure to apply the sigmoid function to them!
            pred_sim = self.assign_net(track_features.cuda(),
                                       pred_features.cuda().cuda(),
                                       track_boxes.cuda(),
                                       boxes.cuda(),
                                       track_t.cuda(),
                                       curr_t.cuda())
            pred_sim = torch.sigmoid(pred_sim)
            pred_sim = pred_sim.cpu().numpy()

            ########################
            #### TODO ends
            ########################

            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = (1 - pred_sim)

            # Do not allow mataches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)


        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)


import motmetrics as mm
from tracker.utils import get_mot_accum


def evaluate_mot_accums_own(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)

    # get idf1 and mota
    idf_v = summary['idf1']['OVERALL']
    mota_v = summary['mota']['OVERALL']
    return idf_v, mota_v

def run_tracker_own(val_sequences, db, tracker, output_dir=None):
    time_total = 0
    mot_accums = []
    results_seq = {}
    for seq in val_sequences:
        # break
        tracker.reset()
        now = time.time()

        print(f"Tracking: {seq}")

        # data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        with torch.no_grad():
            # for i, frame in enumerate(tqdm(data_loader)):
            for frame in db[str(seq)]:
                tracker.step(frame)

        results = tracker.get_results()
        results_seq[str(seq)] = results

        if seq.no_gt:
            print(f"No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        time_total += time.time() - now

        print(f"Tracks found: {len(results)}")
        print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            seq.write_results(results, os.path.join(output_dir))

    print(f"Runtime for all sequences: {time_total:.1f} s.")
    if mot_accums:
        idf_v, mota_v = evaluate_mot_accums_own(mot_accums, [str(s) for s in val_sequences if not s.no_gt],
                                                generate_overall=True)
        return idf_v, mota_v
    return None, None


if __name__ == '__main__':

    from gnn.dataset import LongTrackTrainingDataset
    from torch.utils.data import DataLoader
    from gnn.trainer import train_one_epoch

    train_db = torch.load(osp.join(gnn_root_dir, 'data/preprocessed_data_train_2.pth'))

    best_idf = None
    EARLY_ST_INIT = 8
    EARLY_ST = EARLY_ST_INIT

    MAX_PATIENCE = 20
    MAX_EPOCHS = 20
    EVAL_FREQ = 1

    assign_net = AssignmentSimilarityNet(reid_network=None,  # Not needed since we work with precomputed features
                                         node_dim=32,
                                         edge_dim=64,
                                         reid_dim=512,
                                         edges_in_dim=6,
                                         num_steps=10).cuda()
                                         #num_steps = 10, dropout=0.2).cuda()#TODO
    # We only keep two sequences for validation. You can
    dataset = LongTrackTrainingDataset(dataset='MOT16-train_wo_val2',
                                       db=train_db,
                                       root_dir=osp.join(root_dir, 'data/MOT16'),
                                       max_past_frames=MAX_PATIENCE,
                                       vis_threshold=0.25)

    dataset.seq_frames = dataset.seq_frames[:8]  # todo

    data_loader = DataLoader(dataset, batch_size=8, collate_fn=lambda x: x,#TODO
                             shuffle=False, num_workers=0, drop_last=True)#TODO
    device = torch.device('cuda')
    optimizer = torch.optim.Adam(assign_net.parameters(), lr=0.0005)  # 0.001#0.0015
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)  # 5

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"-------- EPOCH {epoch:2d} --------")
        train_one_epoch(model=assign_net, data_loader=data_loader, optimizer=optimizer, print_freq=100)
        scheduler.step()

        if epoch % EVAL_FREQ == 0:
            tracker = MPNTracker(assign_net=assign_net.eval(), obj_detect=None, patience=MAX_PATIENCE)
            val_sequences = MOT16Sequences('MOT16-val2', osp.join(root_dir, 'data/MOT16'), vis_threshold=0.)
            idf_v, mota_v = run_tracker_own(val_sequences, db=train_db, tracker=tracker, output_dir=None)
            run_tracker(val_sequences, db=train_db, tracker=tracker, output_dir=None)
            if best_idf == None or idf_v > best_idf:
                best_idf = idf_v
                print("best idf is: " + str(best_idf))
                torch.save(assign_net.state_dict(), root_dir + '/models/assign_net_best')
                EARLY_ST = EARLY_ST_INIT
            else:
                EARLY_ST -= 1
                if EARLY_ST == 0:
                    print("!!!!!!\nMaking early stop\n!!!!!!")
                    break;

