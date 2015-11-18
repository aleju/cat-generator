require 'paths'
require 'nn'
require 'cutorch'
require 'cunn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'stn'

OPT = lapp[[
  --save                (default "logs")                subdirectory in which the model is saved
  --network             (default "adversarial.net")     name of the model file
]]

local filepath = paths.concat(OPT.save, OPT.network)
local tmp = torch.load(filepath)
if tmp.epoch then print("") print("Epoch:") print(tmp.epoch) end
if tmp.opt then print("") print("OPT:") print(tmp.opt) end
if tmp.G then print("") print("G:") print(tmp.G) end
if tmp.G1 then print("") print("G1:") print(tmp.G1) end
if tmp.G2 then print("") print("G2:") print(tmp.G2) end
if tmp.G3 then print("") print("G3:") print(tmp.G3) end
if tmp.D then print("") print("D:") print(tmp.D) end
