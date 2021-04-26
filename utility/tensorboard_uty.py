from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np 
import matplotlib.pyplot as plt
 



class TensorboardWorker:
    def __init__(self, dir):
        self.dir = dir 
        self.writer = SummaryWriter(dir)

    # helper functions

    def images_to_probs(self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy()) # preds are the index, and output is the logits value
        return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def accuracy(self, net, images, labels):
        self.preds, self.probs = self.images_to_probs(net, images)
        acc = [1 if i==j else 0 for i, j in zip(self.preds, labels)]
        return np.mean(acc)
    
    def plot_classes_preds(self, net, images, labels, classes):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.preds, self.probs
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4): # just choose 4 out of 100 images as examples to show
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            self.matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig

    