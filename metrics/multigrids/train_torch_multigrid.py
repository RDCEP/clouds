# _*_ coding: utf-8   _*_

#######################################################################
# 
#       Multigrid Residual Autoencoder with Rotate Invariant Loss
# 
#######################################################################
#
#   + Document 
# Test rotate invariant loss in the multi-grid residual autoencode  
#
#   + Version
#   0.0 08/10/2019 David Lorell    Code original functions for the model
#   0.1 08/19/2019 Takuya Kurihana Re-build MGAE from David's from Tak.py by pytorch
#   0.2 08/20/2019 Takuya Kurihana Add PyramidDataset class and channel_last option
#
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from collections import OrderedDict


class TrainData():

    # This is merely a repository for data collected during training/testing.

    def __init__(self):
        super(TrainData, self).__init__()
        self.avgTrainLosses = []
        self.avgTestLosses = []
        self.avgTrainAccuracies = []
        self.avgTestAccuracies = []
        self.alphas = []
        self.numEpochsTrained = 0
        self.timeSpentTraining = 0
        self.modelType = ""
        self.lr = 0
        self.threshold = 0
        self.bestTestAccuracy = -1


class ConvUnit(nn.Module):
    # Convolutional Unit
    # Wrapper around a Batchnorm, ReLU, and one Convolutional Layer.
    def __init__(self, channelsIn, channelsOut, kernel_size, padding):
        super(ConvUnit, self).__init__()
        self.channelsIn = channelsIn
        self.convUnit = nn.Sequential(
            nn.BatchNorm2d(channelsIn),
            nn.ReLU(True),
            nn.Conv2d(channelsIn, channelsOut, kernel_size=kernel_size, padding=padding)
        )
    def forward(self, x):
        return self.convUnit(x)


class MGUnit(nn.Module):
    # Multigrid Unit.
    # The basic multi-grid convolutional layer. Takes a python list of resolution sheets as input.
    #
    # channelsIn=(..., c4, c3, c2, c1, c0)  => The input channel depths for each resolution sheet is ...,c4,c3,c2,c1,c0
    #
    # channelsOut=(...,c4,c3,c2,c1,c0)      => The output channel depths for each resolution sheet is ...,c4,c3,c2,c1,c0
    #
    # sheetResolutions=(a,b,c)              => There are 3 resolutions sheets in each pyramid, and their resolutions,
    #                                          respectively, are a, b, and c.

    def __init__(self, channelsIn, channelsOut, kernel_size, padding, sheetResolutions=(32, 16, 8, 4, 2)):
        super(MGUnit, self).__init__()
        self.sheetResolutions = sheetResolutions
        self.channelsIn = channelsIn
        self.channelsOut = channelsOut

        convs = []
        for i, res in enumerate(sheetResolutions):
            if channelsOut[i] == 0:
                convs.append(None)
                continue

            if i == 0:
                inChannels = channelsIn[i] + channelsIn[i+1]
            elif i == (len(sheetResolutions) - 1):
                inChannels = channelsIn[i-1] + channelsIn[i]
            else:
                inChannels = channelsIn[i - 1] + channelsIn[i] + channelsIn[i + 1]

            if i == len(sheetResolutions) -1:
                k_size = 1
                pad = 0
            else:
                k_size = kernel_size
                pad = padding

            convs.append(ConvUnit(inChannels, channelsOut[i], kernel_size=k_size, padding=pad))

        self.mgconvs = nn.ModuleList(convs)
        self.shrink = nn.MaxPool2d(2, 2)
        self.grow = nn.Upsample(scale_factor=2)

    def forward(self, x):
        if not(len(self.sheetResolutions) == len(x)):
            print("The data isn't the right shape. Should have a place (np.array([]) at least) for each "
                  "resolution level.")
            print("Expected resolution levels: ", self.sheetResolutions)
            print("Number of sheets in input: ", len(x))
            print("Number of sheets expected: ", len(self.sheetResolutions))
            print(x[0].size())
            exit(0)

        # Figure out the batch size. Important for the last step.
        batchSize = -1
        for i in x:
            if i.size()[0] == 0:
                continue
            else:
                batchSize = i.size()[0]

        # First duplicate, resize, and concatenate the appropriate sheets of the input pyramids.
        processedPyramid = self.preprocessInput(x, batchSize)

        # Now send the concatenated results through the appropriate conv layers
        outPyramid = []
        for i, resolutionSheet in enumerate(processedPyramid):
            if not(self.mgconvs[i]):
                outPyramid.append(torch.from_numpy(np.empty((batchSize, 0))))
                continue

            outPyramid.append(self.mgconvs[i](resolutionSheet))

        return outPyramid

    def down(self, x, degree=2):
        ogres = x.size()[-1]
        while ogres/x.size()[-1] < degree:
            x = self.shrink(x)
        return x

    def up(self, x, degree=2):
        ogres = x.size()[-1]
        while x.size()[-1]/ogres < degree:
            x = self.grow(x)
        return x

    def preprocessInput(self, x, batchSize):
        processedPyramid = []

        for i, resolutionSheet in enumerate(x):
            if not (self.mgconvs[i]):
                # No outchannels for this resolution.
                processedPyramid.append(torch.from_numpy(np.empty((batchSize, 0))))
                continue

            inputisValid = True
            if x[i].size()[1] == 0:
                # The input has no valid sheet at this resolution
                inputisValid = False

            fromAbove = None
            fromBelow = None
            allChannels = []
            # We are in a middling resolution
            if i > 0 and i < (len(x) - 1):

                # Factor by which to downsample images across these sheets.
                downDegree = self.sheetResolutions[i - 1] // self.sheetResolutions[i]
                # Factor by which to upsample images across these sheets.
                upDegree = self.sheetResolutions[i] // self.sheetResolutions[i + 1]

                # There is a valid sheet below us
                if x[i - 1].size()[1] > 0:
                    fromBelow = self.down(x[i - 1], downDegree)
                # There is a valid sheet above us
                if x[i + 1].size()[1] > 0:
                    fromAbove = self.up(x[i + 1], upDegree)

                # Collect the channels from all valid sheets.
                if isinstance(fromBelow, torch.Tensor):
                    allChannels.append(fromBelow)
                if inputisValid:
                    allChannels.append(resolutionSheet)
                if isinstance(fromAbove, torch.Tensor):
                    allChannels.append(fromAbove)

            # We are at the base resolution
            elif i == 0:
                # Factor by which to upsample images across these sheets.
                upDegree = self.sheetResolutions[i] // self.sheetResolutions[i + 1]

                # There is a valid sheet above us
                if x[i + 1].size()[1] > 0:
                    fromAbove = self.up(x[i + 1], upDegree)

                # Collect the channels from all valid sheets.
                if inputisValid:
                    allChannels.append(resolutionSheet)
                if isinstance(fromAbove, torch.Tensor):
                    allChannels.append(fromAbove)

            # We are at the poorest resolution, at the top of the pyramid.
            elif i == len(x) - 1:
                # Factor by which to downsample images across these sheets.
                downDegree = self.sheetResolutions[i - 1] // self.sheetResolutions[i]

                # There is a valid sheet below us
                if x[i - 1].size()[1] > 0:
                    fromBelow = self.down(x[i - 1], downDegree)

                # Collect the channels from all valid sheets.
                if isinstance(fromBelow, torch.Tensor):
                    allChannels.append(fromBelow)
                if inputisValid:
                    allChannels.append(resolutionSheet)

            # We have gone wrong somewhere. God help us all.
            else:
                print("There are no more resolutions. Should not have gotten here.")
                exit(0)

            # What if there were no valid sheets above or below us? Then we just deal with whatever we already had.
            if len(allChannels) == 0:
                processedSheet = resolutionSheet
            # Otherwise, bring it all together!
            else:
                processedSheet = torch.cat(allChannels, dim=1)

            # Now add the single contiguous sheet to the pyramid for later convolving.
            processedPyramid.append(processedSheet)

        return processedPyramid


class MGResUnit(nn.Module):

    # Multigrid Residual Unit.
    #
    # This class pushes the input data through two MGUnit layers and sums the output with the original input at the end.
    #
    # residuals=False  =>  No residual connection will be made.
    #
    # noInputRes=True  =>  The residual connection is made between the output of the first MGUnit and the output of the
    #                      second MGUnit. No skip connection from the original input.
    #
    # noOutputRes=True =>  The residual connection is made between the original input and the output of the first
    #                      MGUnit. No skip connection to the final output.
    #
    # backwards=True   =>  The increase/decrease in channel depth will occur first as opposed to last.
    #
    # sheetResolutions=(a,b,c)              => There are 3 resolutions sheets in each pyramid, and their resolutions,
    #                                          respectively, are a, b, and c.

    def __init__(self, channelsIn, channelsOut, kernel_size, padding, sheetResolutions=(32, 16, 8, 4, 2),
                 residuals=True, noInputRes=False, noOutputRes=False, backwards=False):
        super(MGResUnit, self).__init__()
        self.residuals = residuals
        self.noInputRes = noInputRes
        self.noOutputRes = noOutputRes

        if backwards:
            self.MG0 = MGUnit(channelsIn, channelsIn, kernel_size, padding, sheetResolutions=sheetResolutions)
            self.MG1 = MGUnit(channelsIn, channelsOut, kernel_size, padding, sheetResolutions=sheetResolutions)
        else:
            self.MG0 = MGUnit(channelsIn, channelsOut, kernel_size, padding, sheetResolutions=sheetResolutions)
            self.MG1 = MGUnit(channelsOut, channelsOut, kernel_size, padding, sheetResolutions=sheetResolutions)

    def forward(self, x):
        if self.noInputRes:
            x = self.MG0(x)
            convOut = self.MG1(x)
        elif self.noOutputRes:
            convOut = self.MG0(x)
        else:
            convOut = self.MG1(self.MG0(x))

        outPyramid = []
        for i, resolutionSheet in enumerate(convOut):
            if self.residuals:
                # If there are any channels carrying information in the previous resolutionSheet, make a skip connection
                # between them and their corresponding channels in the new one.
                if resolutionSheet.size()[1] > 0 and x[i].size()[1] > 0:
                    channelsIn = x[i].size()[1]
                    channelsOut = resolutionSheet.size()[1]

                    if channelsOut == channelsIn:
                        finalConv = x[i] + resolutionSheet
                    elif channelsOut < channelsIn:
                        finalConv = x[i][:, :channelsOut, :, :] + resolutionSheet
                    else:
                        c1 = resolutionSheet[:, :channelsIn, :, :] + x[i]
                        c2 = resolutionSheet[:, channelsIn:, :, :]
                        finalConv = torch.cat((c1, c2), dim=1)
                else:
                    finalConv = resolutionSheet
            else:
                finalConv = resolutionSheet

            outPyramid.append(finalConv)

        if self.noOutputRes:
            ret = self.MG1(outPyramid)
        else:
            ret = outPyramid

        return ret


class MGSeries(nn.Module):

    # Multigrid Series
    #
    # This class implements an easy way to get a series of MGUnits wihtout residual connections.
    #
    # The arguments are the same as those for the MGUnit class, though missing the "padding" argument, as
    # it is assumed that the resolution dimensions will be maintained throughout the sequence.
    #
    # Two additional arguments:
    #
    # units=n  => n MGUnits will be built into a sequence.
    #
    # backwards=True => The change in channel depth will be accomplished first rather than last.
    #
    # sheetResolutions=(a,b,c)              => There are 3 resolutions sheets in each pyramid, and their resolutions,
    #                                          respectively, are a, b, and c.

    def __init__(self, inChannels, outChannels, kernel_size, sheetResolutions, units=1, backwards=False):
        super(MGSeries, self).__init__()

        padcheck = (kernel_size - 1)
        if padcheck / 2 != padcheck // 2:
            print(kernel_size)
            print("Trying to make an MGSeries which is unable to preserve feature size. Choose more carefully.")
            exit(1)
        padding = padcheck // 2

        if units == 1:
            self.mgseries = MGUnit(inChannels, outChannels, kernel_size, padding, sheetResolutions=sheetResolutions)
        else:
            layers = OrderedDict()
            if backwards:
                for i in range(0, units-1):
                    layers[str(i)] = MGUnit(inChannels, inChannels, kernel_size, padding, sheetResolutions=sheetResolutions)
                layers[str(units-1)] = MGUnit(inChannels, outChannels, kernel_size, padding, sheetResolutions=sheetResolutions)
            else:
                layers["0"] = MGUnit(inChannels, outChannels, kernel_size, padding, sheetResolutions=sheetResolutions)
                for i in range(1, units):
                    layers[str(i)] = MGUnit(outChannels, outChannels, kernel_size, padding, sheetResolutions=sheetResolutions)
            self.mgseries = nn.Sequential(layers)

    def forward(self, x):
        return self.mgseries(x)


class MGResSeries(nn.Module):

    # Multigrid Residual Series
    #
    # This class implements an easy way of obtaining a sequence of MGResUnits of the same resolution.
    #
    # The arguments are the same as for the MGSeries, except there is no opportunity for changing the channel depth
    # within the class, and correspondingly there is no need for a "backwards" argument.
    #
    # sheetResolutions=(a,b,c)              => There are 3 resolutions sheets in each pyramid, and their resolutions,
    #                                          respectively, are a, b, and c.

    def __init__(self, channels, kernel_size, sheetResolutions, units=1):
        super(MGResSeries, self).__init__()

        padcheck = (kernel_size - 1)
        if padcheck / 2 != padcheck // 2:
            print(kernel_size)
            print("Trying to make an MGSeries which is unable to preserve feature size. Choose more carefully.")
            exit(1)
        padding = padcheck // 2

        layers = OrderedDict()
        for i in range(0, units):
            layers[str(i)] = MGResUnit(channels, channels, kernel_size, padding, sheetResolutions=sheetResolutions)

        self.mgresseries = nn.Sequential(layers)

    def forward(self, x):
        return self.mgresseries(x)

class PyramidDataset(torch.utils.data.Dataset):
    def __init__(self, inputData, targetData, layerReses=(32, None, None, None, None), transform=None, channel_last=True):
        super(PyramidDataset, self).__init__()
        
        print("\nPreparing the data...")
        
        self.targets = targetData.targets
        self.transform = transform
        self.pool = nn.MaxPool2d(2, 2)
        self.layerReses = layerReses
        self.channel_last = channel_last
        if torch.cuda.is_available():
          self.data = [resolution.cuda() for resolution in self.makePyramid(inputData.data)]
        else:
          self.data = [resolution.cpu() for resolution in self.makePyramid(inputData.data)]

    def makePyramid(self, x):
        # add t.kurihana
        if self.channel_last:
          x = np.rollaxis(x,-1,1)          

        ogres = x.shape[-2]
        pyramid = []
        #x = x.astype('uint8')
        images = []
        for i, elt in enumerate(x):
            images.append(self.transform(x[i]))
        if torch.cuda.is_available():
          x = torch.stack(images).cuda()
        else:
          x = torch.stack(images).cpu()

        for res in self.layerReses:
            if not(isinstance(res, int)):
                pyramid.append(torch.from_numpy(np.array([])))
                continue
            degree = ogres / res
            sheet = self.subsample(x, degree)
            pyramid.append(sheet)
            
        return pyramid

    def subsample(self, x, degree=2):
        if not (degree % 2 == 0) and not (degree == 1):
            print("Please choose a power of 2 for the pyramidal resolution changes.")
            exit(0)
        k = 1
        while k < degree:
            k = k * 2
            x = self.pool(x)
        return x

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        unprocessedTarget = self.targets[idx]
        unprocessedSheets = []
        for resolution in self.data:
            if resolution.size()[0] > 0:
                unprocessedSheets.append(resolution[idx])
            else:
                unprocessedSheets.append(resolution)


        sample = (unprocessedSheets, unprocessedTarget)

        return sample


class MGAutoencoder(nn.Module):

    # Multigrid Autoencoder
    #
    # This class implements the multigrid residual autoencoder.
    #
    # units=n  => If n > 0: Create an autoencoder composed of two MGSeries with n units in each. No skip connections.
    #             If n < 0: Create an autoencoder composed of an encoder and a decoder where the encoder is composed of
    #                       1 MGUnit which transforms the input channels into the specified channels below, followed by
    #                       1 MGResSeries with as many units as specified below, followed by 3 MGUnits which gradually
    #                       reduce the channel depth of all except the top two resolution sheets to 0.
    #
    # BE SURE TO NOTICE THE HARD-CODED VARIABLES DEFINED UNDERNEATH "---------- ATTENTION ----------"
    # These determine the architecture of the autoencoder. Sorry that they are not passable arguments yet.

    def __init__(self, units=5):
        super(MGAutoencoder, self).__init__()
        self.trainData = TrainData()
        self.trainData.modelType = "MGAutoencoderRes"+str(units)
        self.output_middle_features = False

        if units < 0:
            kernel_size = 3
            padcheck = (kernel_size - 1)
            if padcheck / 2 != padcheck // 2:
                print(kernel_size)
                print("Trying to make an MGSeries which is unable to preserve feature size. Choose more carefully.")
                exit(1)
            padding = padcheck // 2

            # -------------------------------  ATTENTION ---------------------------------------------------
            #
            #   The variables below here are those that you should change. "c" is the base number of channels for each
            #   resolution sheet. By default, as you can see below, the lowest two resolution sheets
            #   have 16*3 and 16*4 channels assigned to them by default. This is a quirk of my own needs and they may be
            #   changed at will. They also define the bottleneck, as the bottleneck features are extracted in the
            #   forward() method by reshaping the lowest 2 resolution sheets into a single _x2x2 tensor.
            #


            c = -units
            c4 = c
            c3 = c
            c2 = c
            c1 = 16 * 3
            c0 = 16 * 4
            sheetResolutions = (32, 16, 8, 4, 2)
            units = 16


            self.encoder = nn.Sequential(
                MGUnit((3,3,3,3,3), (c4, c3, c2, c1, c0), 3, padding, sheetResolutions=sheetResolutions),
                MGResSeries((c4, c3, c2, c1, c0), 3, sheetResolutions=sheetResolutions, units=units),
                MGResUnit((c4, c3, c2, c1, c0), (0, c3, c2, c1, c0), kernel_size, padding, sheetResolutions=sheetResolutions),
                MGResUnit((0, c3, c2, c1, c0), (0, 0, c2, c1, c0), kernel_size, padding, sheetResolutions=sheetResolutions),
                MGResUnit((0, 0, c2, c1, c0), (0, 0, 0, c1, c0), kernel_size, padding, sheetResolutions=sheetResolutions, noOutputRes=True)
            )
            self.decoder = nn.Sequential(
                MGResUnit((0, 0, 0, c1, c0), (0, 0, c2, c1, c0), kernel_size, padding, sheetResolutions=sheetResolutions, noInputRes=True, backwards=True),
                MGResUnit((0, 0, c2, c1, c0), (0, c3, c2, c1, c0), kernel_size, padding, sheetResolutions=sheetResolutions, backwards=True),
                MGResUnit((0, c3, c2, c1, c0), (c4, c3, c2, c1, c0), kernel_size, padding,sheetResolutions=sheetResolutions, backwards=True),
                MGResSeries((c4, c3, c2, c1, c0), 3, sheetResolutions=sheetResolutions, units=units),
                MGUnit((c4, c3, c2, c1, c0), (3, 0, 0, 0, 0), 3, padding, sheetResolutions=sheetResolutions),
            )

        else:
            self.encoder = nn.Sequential(
                # Printer(),
                MGSeries((3, 3, 3, 3, 3), (8, 8, 8, 8, 8), 3, sheetResolutions=(32, 16, 8, 4, 2), units=units)
            )
            self.decoder = nn.Sequential(
                # Printer(),
                MGSeries((8, 8, 8, 8, 8), (3, 3, 3, 3, 3), 3, sheetResolutions=(32, 16, 8, 4, 2), units=units, backwards=True)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        if self.output_middle_features:
            # Concat the features of the 4x4 sheet with the features of the 2x2 sheet.
            reshaped = features[-2].view(features[-2].size()[0], -1, 2, 2)
            allFeatures = torch.cat((features[-1], reshaped), dim=1)
            return self.sigmoid(allFeatures)
        else:
            reconstruction = self.decoder(features)
            # Grabs the highest resolution sheet from the reconstructed pyramid.
            return self.sigmoid(reconstruction[0])


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument(
    '--datadir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--savedir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--batch_size',
    type=int,
    default=32
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.001
  )
  p.add_argument(
    '--num_epoch',
    type=int,
    default=10
  )
  p.add_argument(
    '--save_every',
    type=int,
    default=10
  )
  p.add_argument(
    '--units',
    type=int,
    default=-1,
    help=' set negative number to use residual connection'
  )
  args = p.parse_args()
  for f in args.__dict__:
    print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
  print("\n")
  return args

if __name__ == "__main__":
  # set params
  FLAGS = get_args()

  # set directory
  os.makedirs(FLAGS.savedir, exist_ok=True)

  # check device(CPU or GPU) 
  pin_memory=False
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
    #TODO
    #pin_memory=True

  # set dataset
  # train dataset
  #   TODO: add num_workers 
  # get original dataset 
  train_dataset = datasets.CIFAR10(
      root="../multigrids/cifar_data",
      train=True,
      download=False
  )
  # convert original dataset to pyramid dataset with transforms
  # TODO: add normalization
  #
  CIFAR10dataset = PyramidDataset(train_dataset,train_dataset, 
                 layerReses=(32, 16, 8, 4, 2), 
                 transform=transforms.Compose([
                   transforms.ToPILImage(),
                   transforms.Resize((32,32)),
                   transforms.ToTensor()
                 ]),
                 channel_last=True
  ) 

  train_loader = torch.utils.data.DataLoader(
    CIFAR10dataset,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    pin_memory=pin_memory
  )
  #TODO add test dataset for inference 
  # test dataset
  #test_loader = torch.utils.data.DataLoader(
  #  datasets.MNIST(
  #    root=FLAGS.datadir,
  #    train=False,
  #    transform=transforms.Compose([
  #      transforms.ToTensor(),
  #      transforms.Normalize((0.5,), (0.5,))
  #    ])    
  #  ),
  #  batch_size=FLAGS.batch_size,
  #  shuffle=True,
  #  pin_memory=True
  #)

  # load model
  model = MGAutoencoder(FLAGS.units)  
  model = model.to(device)  

  # set optimizer and loss
  # TODO: add params for Adam
  optimizer = torch.optim.Adam(model.parameters(),lr=FLAGS.lr) 
  # BCELoss = Binary Cross Entropy Loss
  criterion = nn.BCELoss().cuda() if torch.cuda.is_available() else nn.BCELoss()
  # TODO add custom loss here

  #==========================================================
  #                 Training Process
  #==========================================================

  stime = time.time()
  for epoch in range(FLAGS.num_epoch):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
      # forward pass:
      #output = model(inputs)
      encoded_img = model.encoder(inputs)
      decoded_img = model.decoder(encoded_img)
      for idx, (idecoded_img, iencoded_img) in enumerate(zip(decoded_img, encoded_img) ):
        print("layer {}".format(idx))
        print('encoder', iencoded_img.cpu().detach().numpy().shape, flush=True)
        print('decoder', idecoded_img.cpu().detach().numpy().shape, flush=True)
      
      #compute loss 
      #input_imgs = torch.cat([i.cuda() for i in inputs])
      #input_imgs = [i.cuda() for i in inputs][0] # get data corresponding to output?
      #print('input shape', input_imgs.cpu().detach().numpy().shape, flush=True)
      #print('output size', output.cpu().detach().numpy().size, flush=True)
      #print('output shape', output.cpu().detach().numpy().shape, flush=True)
      loss = criterion(decoded_img, inputs)
      stop
      running_loss += loss

      # zero gradients, perform a backward pass, and update weights
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # show intermediate learning result
    print( " ### Num. of Epochs {}  | Loss {} ### ".format(
        epoch,running_loss/float(FLAGS.batch_size) ), flush=True )

    if epoch % FLAGS.save_every == 0:
      # save model
      # TODO: change the oname 
      oname = 'mgrencoder_'+str(FLGAS.num_epoch)+'.pth'
      torch.save(
          model.state_dict(), 
          os.path.join(FLAGS.savedir, oname)
      )
  
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
