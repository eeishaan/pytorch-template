#!/usr/bin/env python3
# Copyright (C) 2019 Ishaan Kumar
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from types import SimpleNamespace

import torch

from skeletal.cfg import DEVICE


class BaseExperiment(object):
    '''
    Base class for experiments
    '''

    def __init__(
            self,
            experiment_dir,
            model_obj,
            summary_writer=None,
            optimizer=None,
            criterion=None,
            **params
    ):
        self._experiment_file = experiment_dir / 'model.pt'
        self._model_obj = model_obj
        self._optimizer = optimizer
        self._criterion = criterion
        self._summary_writer = summary_writer
        # send model to device
        self._model_obj.to(DEVICE)
        if summary_writer is not None:
            self._summary_writer.add_text(
                'model_obj', repr(self._model_obj), 0)
        # initialize epoch var correctly
        self._start_epoch = 0

        for k, v in params.items():
            setattr(self, k, v)

    def before_test(self, ctx):
        ctx.predictions = []

    def before_test_forwardp(self, ctx, data):
        return data

    def after_test_forwardp(self, ctx, outputs):
        pass

    def after_test(self, ctx):
        pass

    def before_train_epoch(self, ctx, train_train_no_aug_loader):
        pass

    def before_train_forwardp(self, ctx, data):
        return data

    def before_train_backprop(self, ctx, outputs, data):
        pass

    def after_train_backprop(self, ctx, outputs, data):
        pass

    def after_train_epoch(self, ctx):
        epoch = ctx.epoch

        # save embedding model after 10 epochs
        if epoch % 10 != 9:
            self.save_experiment(ctx)

        # print loss
        loss = ctx.running_loss.item()
        message = "Epoch: {} Train Loss: {}".format(epoch, loss)
        print(message)
        self._summary_writer.add_scalar('train_loss', loss, epoch)
        # return True as we don't want to stop
        return True

    def compute_loss(self, ctx, outputs, labels):
        return self._criterion(outputs, labels)

    def test(self, dataloader):
        self._model_obj.eval()
        ctx = SimpleNamespace()
        self.before_test(ctx)
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                data = self.before_test_forwardp(ctx, data)
                data = data.to(DEVICE)
                output = self._model_obj(data)
                self.after_test_forwardp(ctx, output)
        return self.after_test(ctx)

    def train(self, train_loader, valid_loader, epochs, start_epoch=None):
        start_epoch = start_epoch \
            if start_epoch is not None \
            else self._start_epoch

        for epoch in range(start_epoch, epochs):
            self._model_obj.train()

            # prepare context for hooks
            ctx = SimpleNamespace(
                epoch=epoch,
                batch=0,
                running_loss=0,
                valid_loader=valid_loader,
            )

            self.before_train_epoch(ctx, train_loader)
            for batch, data in enumerate(train_loader):
                ctx.batch = batch
                data = data.to(DEVICE)

                # before_forwardp can add second layer of transformation
                data = self.before_train_forwardp(ctx, data)

                # zero out previous gradient
                self._model_obj.zero_grad()

                outputs = self._model_obj(data)
                self.before_train_backprop(ctx, outputs, data)
                loss = self.compute_loss(ctx, outputs, data)
                ctx.running_loss += loss
                loss.backward()
                self._optimizer.step()

                self.after_train_backprop(ctx, outputs, data)

            # Divide the loss by the number of batches
            ctx.running_loss /= len(train_loader)
            is_continue = self.after_train_epoch(ctx)
            if not is_continue:
                break

    def load_experiment(self):
        checkpoint = torch.load(self._experiment_file)
        self._model_obj.load_state_dict(
            checkpoint['model_state_dict'])
        if self._optimizer is not None and \
                'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
        self._start_epoch = checkpoint.get('epoch', 0)

    def save_experiment(self, ctx):
        save_dict = {
            'epoch': ctx.epoch,
            'model_state_dict': self._model_obj.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        torch.save(save_dict, self._experiment_file)
