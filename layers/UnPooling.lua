-- modified version of https://github.com/willwhitney/dc-ign/blob/master/modules/UnPooling.lua
local UnPooling, parent = torch.class('nn.UnPooling', 'nn.Module')

require 'sys'
require 'cutorch'

function UnPooling:__init(s)
   parent.__init(self)

   self.scale = s

   self.indices = torch.Tensor()
end

function UnPooling:updateOutput(input)
   input = input:float()
   local nbBatch = input:size()[1]
   local nbChannels = input:size()[2]
   
   local height = input:size()[3]
   local width = input:size()[4]
   local heightUnpooled = height * self.scale
   local widthUnpooled = width * self.scale
   
   self.output = torch.zeros(nbBatch, nbChannels, heightUnpooled, widthUnpooled)
   
   local ii = 1
   local jj = 1

   self.mapping = {} -- store non-zero mappings for gradient calc

   for i=1,heightUnpooled,self.scale do
      jj = 1;
      for j=1,widthUnpooled,self.scale do
         self.output[{{},{},i,j}] = input[{{},{}, ii,jj}]
         self.mapping[ii ..jj] = {i,j}
         jj = jj + 1;
      end
      ii = ii + 1;
   end

   self.output = self.output:cuda()
   return self.output
end

function UnPooling:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:float()
   input = input:float()

   local nbBatch = input:size()[1]
   local nbChannels = input:size()[2]
   
   local height = input:size()[3]
   local width = input:size()[4]
   local heightUnpooled = height * self.scale
   local widthUnpooled = width * self.scale
   
   self.gradInput = torch.zeros(nbBatch, nbChannels, height, width)

   for ii=1,height do
      for jj=1,width do
         local t = self.mapping[ii .. jj]
         i = t[1]; j = t[2];
         self.gradInput[{{},{},ii,jj}] = gradOutput[{{},{}, i,j}]   
      end
   end

   self.gradInput = self.gradInput:cuda()
   return self.gradInput
end
