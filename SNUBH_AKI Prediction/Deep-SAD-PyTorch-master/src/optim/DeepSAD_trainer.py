from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, alpha: float=1.0, fairness_type: str='EO', optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):

        """
        Inits DeepSADTrainer with added Fairness parameter alpha.

        Args:
            c: Hypersphere center.
            eta: Deep SAD hyperparameter.
            alpha: Weight for fairness loss.
        """

        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Fairness parameters
        self.alpha = alpha
        self.fairness_type = fairness_type

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def demographic_parity_loss(self, dist, sensitive_attr):
        """
        Computes the Demographic Parity loss to ensure fairness.

        Args:
           dist: Distance between output and center.
           sensitive_attr: Sensitive attribute.

        Returns:
           fairness_loss: Fairness loss based on Demographic Parity.
        """

        # Calculate positive rates for each sensitive group
        male_idx = (sensitive_attr == 0)
        female_idx = (sensitive_attr == 1)

        male_scores = dist[male_idx]
        female_scores = dist[female_idx]

        if male_scores.numel() == 0 or female_scores.numel() == 0:
            return torch.tensor(0.0).to(self.device)

        # Threshold can be set as the mean distance
        threshold = dist.mean()

        male_positive_rate = (male_scores > threshold).float().mean()
        female_positive_rate = (female_scores > threshold).float().mean()

        fairness_loss = torch.abs(male_positive_rate - female_positive_rate)

        return fairness_loss

    def equity_odds_loss(self, outputs, sensitive_attr, labels, dist):
        """
        Computes the Equity Odds loss to ensure fairness.

        Args:
            outputs: Model outputs.
            sensitive_attr: Sensitive attribute
            labels: True target values.
            dist: Distance between output and center.

        Returns:
            fairness_loss: Fairness loss based on equity odds.
        """

        # Select only labeled data (labels 0 or 1)
        known_idx = (labels == 0) | (labels == 1)
        if known_idx.sum() == 0:
            return torch.tensor(0.0).to(self.device)

        outputs = outputs[known_idx]
        sensitive_attr = sensitive_attr[known_idx]
        labels = labels[known_idx]
        dist = dist[known_idx]

        # Split by sensitive attribute
        male_idx = (sensitive_attr == 0)
        female_idx = (sensitive_attr == 1)

        # Initialize fairness_loss
        fairness_loss = torch.tensor(0.0).to(self.device)

        for y in [0, 1]:
            # For each class
            male_mask = male_idx & (labels == y)
            female_mask = female_idx & (labels == y)

            if male_mask.sum() > 0 and female_mask.sum() > 0:
                male_mean = dist[male_mask].mean()
                female_mean = dist[female_mask].mean()
                fairness_loss += torch.abs(male_mean - female_mean)

        return fairness_loss

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            epoch_fairness_loss = 0.0  # Fairness loss accumulator
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                inputs, labels, semi_targets, _, sensitive_attr = data

                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                sensitive_attr = sensitive_attr.to(self.device)

                # Debugging: Print distribution of semi_targets
                # Uncomment the following lines for debugging purposes
                # unique, counts = torch.unique(semi_targets, return_counts=True)
                # logger.info(f"Epoch {epoch+1}, Batch {n_batches+1}: semi_targets distribution: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # Create masks
                mask_known_normal = (semi_targets == 0)
                mask_known_outlier = (semi_targets == 1)
                mask_unlabeled = (semi_targets == -1)

                # Compute losses
                loss_normal = torch.mean(dist[mask_known_normal]) if mask_known_normal.sum() > 0 else 0.0
                loss_outlier = torch.mean(self.eta * (dist[mask_known_outlier] + self.eps)) if mask_known_outlier.sum() > 0 else 0.0
                loss_unlabeled = torch.mean(dist[mask_unlabeled]) if mask_unlabeled.sum() > 0 else 0.0

                # Total DeepSAD loss
                deep_sad_loss = loss_normal + loss_outlier + loss_unlabeled

                # Calculate fairness loss
                if self.fairness_type == 'EO':
                    fairness_loss = self.equity_odds_loss(outputs, sensitive_attr, labels, dist)
                elif self.fairness_type == 'DP':
                    fairness_loss = self.demographic_parity_loss(dist, sensitive_attr)
                else:
                    raise ValueError(f"Unknown fairness type: {self.fairness_type}")

                # Combined loss
                total_loss = deep_sad_loss + self.alpha * fairness_loss

                # Backward and optimize
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_fairness_loss += fairness_loss.item()
                n_batches += 1

            # Scheduler step: once per epoch
            scheduler.step()

            # Log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} | '
                        f'| Fair Loss ({self.fairness_type}): {epoch_fairness_loss / n_batches:.6f} |')

            if (epoch + 1) in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        epoch_loss = 0.0
        epoch_fairness_loss = 0.0  # Fairness loss accumulator
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 5:
                    # Training-like data (should not happen in test_loader)
                    inputs, labels, semi_targets, idx, sensitive_attr = data
                    semi_targets = semi_targets.to(self.device)  # Not used in test
                elif len(data) == 4:
                    # Test data without semi_targets
                    inputs, labels, idx, sensitive_attr = data
                    semi_targets = None
                else:
                    raise ValueError("Unexpected number of elements in data tuple.")

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)
                sensitive_attr = sensitive_attr.to(self.device)  # Sensitive attributes

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if semi_targets is not None:
                    # Unlikely scenario in test_loader
                    losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                    loss = torch.mean(losses)
                else:
                    # Standard testing loss
                    loss = torch.mean(dist)

                scores = dist

                # Fairness loss calculation
                if self.fairness_type == 'EO':
                    fairness_loss = self.equity_odds_loss(outputs, sensitive_attr, labels, dist)
                elif self.fairness_type == 'DP':
                    fairness_loss = self.demographic_parity_loss(dist, sensitive_attr)
                else:
                    fairness_loss = torch.tensor(0.0).to(self.device)  # Default

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                epoch_fairness_loss += fairness_loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        if idx_label_score:
            try:
                indices, labels, scores = zip(*idx_label_score)
                labels = np.array(labels)
                scores = np.array(scores)
                self.test_auc = roc_auc_score(labels, scores)
            except ValueError as ve:
                logger.error(f'ROC AUC Score computation failed: {ve}')
                self.test_auc = None
        else:
            logger.warning('No test scores available to compute AUC.')
            self.test_auc = None

        # Compute Test Fair Loss (average fairness loss over all batches)
        if n_batches > 0:
            self.test_fair_loss = epoch_fairness_loss / n_batches
        else:
            logger.warning('No batches processed during testing.')
            self.test_fair_loss = 0.0

        # Update results dictionary
        self.results = {
            'test_auc': self.test_auc,
            'test_fair_loss': self.test_fair_loss,
            'test_scores': self.test_scores
        }
        
        # Log results
        if self.test_auc is not None:
            logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        else:
            logger.info('Test AUC: None')

        if self.test_fair_loss is not None:
            logger.info('Test Fair Loss ({}): {:.6f}'.format(self.fairness_type, self.test_fair_loss))
        else:
            logger.info('Test Fair Loss ({}): None'.format(self.fairness_type))

        if n_batches > 0:
            logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        else:
            logger.info('Test Loss: None')

        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                # import pdb; pdb.set_trace()
                inputs,_,_,_,_ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
