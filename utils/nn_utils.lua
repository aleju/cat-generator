require 'torch'

local nn_utils = {}

-- Sets the weights of a layer to random values within a range.
-- @param weights The weights module to change, e.g. mlp.modules[1].weight.
-- @param range Range for the new values (single number, e.g. 0.005)
function nn_utils.setWeights(weights, range)
    weights:randn(weights:size())
    weights:mul(range)
end

-- Initializes all weights of a multi layer network.
-- @param model The nn.Sequential() model with one or more layers
-- @param rangeWeights A range for the new weights values (single number, e.g. 0.005)
-- @param rangeBias A range for the new bias values (single number, e.g. 0.005)
function nn_utils.initializeWeights(model, rangeWeights, rangeBias)
    rangeWeights = rangeWeights or 0.005
    rangeBias = rangeBias or 0.001

    for m = 1, #model.modules do
        if model.modules[m].weight then
            nn_utils.setWeights(model.modules[m].weight, rangeWeights)
        end
        if model.modules[m].bias then
            nn_utils.setWeights(model.modules[m].bias, rangeBias)
        end
    end
end

-- Creates a tensor of N vectors, each of dimension OPT.noiseDim with random values
-- between -1 and +1.
-- @param N Number of vectors to generate
-- @returns Tensor of shape (N, OPT.noiseDim)
function nn_utils.createNoiseInputs(N)
    local noiseInputs = torch.Tensor(N, OPT.noiseDim):float()
    noiseInputs:uniform(-1.0, 1.0)
    return noiseInputs
end

-- Feeds noise vectors into G or AE+G and returns the result.
-- @param noiseInputs Tensor from createNoiseInputs()
-- @param outputAsList Whether to return the images as one list or as a tensor.
-- @returns Either list of images (as returned by G/AE) or tensor of images
function nn_utils.createImagesFromNoise(noiseInputs, outputAsList)
    local images
    local N = noiseInputs:size(1)
    local nBatches = math.ceil(N/OPT.batchSize)
    for i=1,nBatches do
        local batchStart = 1 + (i-1)*OPT.batchSize
        local batchEnd = math.min(i*OPT.batchSize, N)
        local generated = MODEL_G:forward(noiseInputs[{{batchStart, batchEnd}}]):clone()
        if images == nil then
            local img = generated[1]
            images = torch.Tensor(N, img:size(1), img:size(2), img:size(3)):float()
        end
        images[{{batchStart, batchEnd}, {}, {}, {}}] = generated
    end

    if outputAsList then
        local imagesList = {}
        for i=1, images:size(1) do
            imagesList[#imagesList+1] = images[i]:float()
        end
        return imagesList
    else
        return images
    end
end

-- Creates new random images with G or AE+G.
-- @param N Number of images to create.
-- @param outputAsList Whether to return the images as one list or as a tensor.
-- @returns Either list of images (as returned by G/AE) or tensor of images
function nn_utils.createImages(N, outputAsList)
    return nn_utils.createImagesFromNoise(nn_utils.createNoiseInputs(N), outputAsList)
end

-- Sorts images based on D's certainty that they are fake/real.
-- Descending order starts at y=1 (Y_NOT_GENERATOR) and ends with y=0 (Y_GENERATOR).
-- Therefore, in case of descending order, images for which D is very certain that they are real
-- come first and images that seem to be fake (according to D) come last.
-- @param images Tensor of the images to sort.
-- @param ascending If true then images that seem most fake to D are placed at the start of the list.
--                  Otherwise the list starts with probably real images.
-- @param nbMaxOut Sets how many images may be returned max (cant be more images than provided).
-- @return Tuple (list of images, list of predictions between 0.0 and 1.0)
--                                where 1.0 means "probably real"
function nn_utils.sortImagesByPrediction(images, ascending, nbMaxOut)
    local predictions = torch.Tensor(images:size(1), 1)
    local nBatches = math.ceil(images:size(1)/OPT.batchSize)
    for i=1,nBatches do
        local batchStart = 1 + (i-1)*OPT.batchSize
        local batchEnd = math.min(i*OPT.batchSize, images:size(1))
        predictions[{{batchStart, batchEnd}, {1}}] = MODEL_D:forward(images[{{batchStart, batchEnd}, {}, {}, {}}]):clone()
    end

    local imagesWithPreds = {}
    for i=1,images:size(1) do
        table.insert(imagesWithPreds, {images[i], predictions[i][1]})
    end

    if ascending then
        table.sort(imagesWithPreds, function (a,b) return a[2] < b[2] end)
    else
        table.sort(imagesWithPreds, function (a,b) return a[2] > b[2] end)
    end

    resultImages = {}
    resultPredictions = {}
    for i=1,math.min(nbMaxOut,#imagesWithPreds) do
        resultImages[i] = imagesWithPreds[i][1]
        resultPredictions[i] = imagesWithPreds[i][2]
    end

    return resultImages, resultPredictions
end

-- Visualizes the current training status via Display (based on gfx.js) in the browser.
-- It shows:
--   Images generated from random noise (the noise vectors are set once at the start of the
--   training, so the images should end up similar at each epoch)
--   Images that were deemed "good" by D
--   Images that were deemed "bad" by D
--   Original images from the training set (as comparison)
--   If an Autoencoder is defined, it will show the results of that network (before G is applied
--   as refiner).
-- @param noiseInputs The noise vectors for the random images.
-- @returns void
function nn_utils.visualizeProgress(noiseInputs)
    -- deactivate dropout
    nn_utils.switchToEvaluationMode()

    -- Generate a synthetic test image as sanity test
    -- This should be deemed very bad by D
    local sanityTestImage = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):float()
    sanityTestImage:uniform(0.0, 0.50)
    for i=1,OPT.scale do
        for j=1,OPT.scale do
            if i == j then
                sanityTestImage[1][i][j] = 1.0
            elseif i % 4 == 0 and j % 4 == 0 then
                sanityTestImage[1][i][j] = 0.5
            end
        end
    end

    -- Collect original example images from the training set
    local trainImages = TRAIN_DATA[{{1, 50}, {}, {}, {}}]:clone()

    -- Generate images from G based on the provided noiseInputs
    local rndImages = nn_utils.createImagesFromNoise(noiseInputs)

    -- Place the sanity test image and one original image from the training corpus among
    -- the random Images. The first should be deemed bad by D, the latter as good.
    -- Then find good and bad images (according to D) among the randomly generated ones
    -- Note: has to happen before toRgb() as that would change the color space of the images
    local rndImagesClone = rndImages:clone()
    rndImagesClone[rndImagesClone:size(1)-1] = trainImages[1] -- one real face as sanity test
    rndImagesClone[rndImagesClone:size(1)] = sanityTestImage -- synthetic non-face as sanity test
    local goodImages, _ = nn_utils.sortImagesByPrediction(rndImagesClone, false, 50)
    local badImages, _ = nn_utils.sortImagesByPrediction(rndImagesClone, true, 50)

    if rndImages:ne(rndImages):sum() > 0 then
        print(string.format("[nn_utils vizProgress] Generated images contain NaNs"))
    end

    DISP.image(nn_utils.toRgb(rndImages, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="semi-random generated images (after G)"})
    DISP.image(nn_utils.toRgb(goodImages, OPT.colorSpace), {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title="best samples (first is best)"})
    DISP.image(nn_utils.toRgb(badImages, OPT.colorSpace), {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="worst samples (first is worst)"})
    DISP.image(nn_utils.toRgb(trainImages, OPT.colorSpace), {win=OPT.window+4, width=IMG_DIMENSIONS[3]*15, title="original images from training set"})

    nn_utils.saveImagesAsGrid(string.format("%s/images/%d_%05d.png", OPT.save, START_TIME, EPOCH), nn_utils.toRgb(rndImages, OPT.colorSpace), 10, 10, EPOCH)
    nn_utils.saveImagesAsGrid(string.format("%s/images_good/%d_%05d.png", OPT.save, START_TIME, EPOCH), nn_utils.toRgb(goodImages, OPT.colorSpace), 7, 7, EPOCH)
    nn_utils.saveImagesAsGrid(string.format("%s/images_bad/%d_%05d.png", OPT.save, START_TIME, EPOCH), nn_utils.toRgb(badImages, OPT.colorSpace), 7, 7, EPOCH)

    local rndImagesRating = nn_utils.rateWithV(rndImages)
    local goodImagesRating = nn_utils.rateWithV(goodImages)
    local badImagesRating = nn_utils.rateWithV(badImages)
    table.insert(PLOT_DATA, {EPOCH, rndImagesRating, goodImagesRating, badImagesRating})
    print(string.format("<nnutils viz> [V] semiRandom: %.4f, goodImages: %.4f, badImages: %.4f", rndImagesRating, goodImagesRating, badImagesRating))
    DISP.plot(PLOT_DATA, {win=OPT.window+5, labels={'epoch', 'V(semiRandom)', 'V(goodImages)', 'V(badImages)'}, title='Rating by V'})

    -- reactivate dropout
    nn_utils.switchToTrainingMode()
end

function nn_utils.toRgb(images, from)
    local images = nn_utils.toImageTensor(images)
    if from == "rgb" then
        return images
    elseif from == "y" then
        --[[
        local imagesTmp
        if images:size(4) == nil then
            imagesTmp = images:clone()
        else
            imagesTmp = images:clone():squeeze(2)
        end

        local N = imagesTmp:size(1)
        local height = imagesTmp:size(2)
        local width = imagesTmp:size(3)
        --]]
        return torch.repeatTensor(images, 1, 3, 1, 1)
    elseif from == "hsl" then
        local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4)):float()
        for i=1,images:size(1) do
            out[i] = image.hsl2rgb(images[i])
        end
        return out
    elseif from == "yuv" then
        local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4)):float()
        for i=1,images:size(1) do
            out[i] = image.yuv2rgb(images[i])
        end
        return out
    else
        print("[WARNING] unknown color space <from>: '" .. from .. "'")
    end
end

function nn_utils.rgbToColorSpace(images, colorSpace)
    if colorSpace == "rgb" then
        return images
    else
        if colorSpace == "y" then
            local out = torch.Tensor(images:size(1), 1, images:size(3), images:size(4)):float()
            for i=1,images:size(1) do
                out[i] = nn_utils.rgb2y(images[i])
            end
            return out
        elseif colorSpace == "hsl" then
            local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4)):float()
            for i=1,images:size(1) do
                out[i] = image.rgb2hsl(images[i])
            end
            return out
        elseif colorSpace == "yuv" then
            local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4)):float()
            for i=1,images:size(1) do
                out[i] = image.rgb2yuv(images[i])
            end
            return out
        else
            print("[WARNING] unknown color space in rgbToColorSpace: '" .. colorSpace .. "'")
        end
    end
end

-- convert rgb to grayscale by averaging channel intensities
-- https://gist.github.com/jkrish/29ca7302e98554dd0fcb
function nn_utils.rgb2y(im, threeChannels)
    -- Image.rgb2y uses a different weight mixture
    local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
    if dim ~= 3 then
        print('<error> expected 3 channels')
        return im
    end

    -- a cool application of tensor:select
    local r = im:select(1, 1)
    local g = im:select(1, 2)
    local b = im:select(1, 3)

    local z = torch.zeros(1, w, h):float()

    -- z = z + 0.21r
    z = z:add(0.21, r)
    z = z:add(0.72, g)
    z = z:add(0.07, b)

    if threeChannels == true then
        z = torch.repeatTensor(z, 3, 1, 1)
    end

    return z
end

-- Convert a list (table) of images to a Tensor.
-- If the parameter is already a tensor, it will be returned unchanged.
-- @param imageList A non-empty list/table or tensor of images (each being a tensor).
-- @returns A tensor of shape (N, channels, height, width)
function nn_utils.toImageTensor(imageList, forceChannel)
    if imageList.size ~= nil then
        if not forceChannel or (#imageList:size() == 3) then
            return imageList
        else
            -- forceChannel activated and images lack channel dimension
            -- add it
            local tens = torch.Tensor(imageList:size(1), 1, imageList:size(2), imageList:size(3)):float()
            for i=1,imageList:size(1) do
                tens[i][1] = imageList[i]
            end
            return tens
        end
    else
        if forceChannel == nil then
            forceChannel = false
        end

        local hasChannel = (#imageList[1]:size() == 3)

        local tens
        if hasChannel then
            tens = torch.Tensor(#imageList, imageList[1]:size(1), imageList[1]:size(2), imageList[1]:size(3)):float()
        elseif not hasChannel and forceChannel then
            tens = torch.Tensor(#imageList, 1, imageList[1]:size(1), imageList[1]:size(2)):float()
        else
            tens = torch.Tensor(#imageList, imageList[1]:size(1), imageList[1]:size(2)):float()
        end

        for i=1,#imageList do
            if (not hasChannel and forceChannel) then
                tens[i][1] = imageList[i]
            else
                tens[i] = imageList[i]
            end
        end
        return tens
    end
end

function nn_utils.toImageList(imageTensor, forceChannel)
    local tens = nn_utils.toImageTensor(imageTensor, forceChannel)
    local lst = {}
    for i=1,tens:size(1) do
        table.insert(lst, tens[i])
    end
    return lst
end

-- Switch networks to training mode (activate Dropout)
function nn_utils.switchToTrainingMode()
    if MODEL_AE then
        MODEL_AE:training()
    end
    MODEL_G:training()
    MODEL_D:training()
end

-- Switch networks to evaluation mode (deactivate Dropout)
function nn_utils.switchToEvaluationMode()
    if MODEL_AE then
        MODEL_AE:evaluate()
    end
    MODEL_G:evaluate()
    MODEL_D:evaluate()
end

-- Normalize given images, currently to range -1.0 (black) to +1.0 (white), assuming that
-- the input images are normalized to range 0.0 (black) to +1.0 (white).
-- @param data Tensor of images
-- @param mean_ Currently ignored.
-- @param std_ Currently ignored.
-- @return (mean, std), both currently always 0.5 dummy values
function nn_utils.normalize(data, mean_, std_)
    -- Code to normalize to zero-mean and unit-variance.
    --[[
    local mean = mean_ or data:mean(1)
    local std = std_ or data:std(1, true)
    local eps = 1e-7
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        data[i]:add(-1, mean)
        data[i]:cdiv(std + eps)
    end

    return mean, std
    --]]

    -- Code to normalize to range -1.0 to +1.0, where -1.0 is black and 1.0 is the maximum
    -- value in this image.
    --[[
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        local m = torch.max(data[i])
        data[i]:div(m * 0.5)
        data[i]:add(-1.0)
        data[i] = torch.clamp(data[i], -1.0, 1.0)
    end
    --]]

    -- Normalize to range -1.0 to +1.0, where -1.0 is black and +1.0 is white.
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        data[i]:mul(2)
        data[i]:add(-1.0)
        data[i] = torch.clamp(data[i], -1.0, 1.0)
    end

    -- Dummy return values
    return 0.5, 0.5
end

-- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
function nn_utils.zeroDataSize(data)
    if type(data) == 'table' then
        for i = 1, #data do
            data[i] = nn_utils.zeroDataSize(data[i])
        end
    elseif type(data) == 'userdata' then
        data = torch.Tensor():typeAs(data)
    end
    return data
end

-- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
-- Resize the output, gradInput, etc temporary tensors to zero (so that the on disk size is smaller)
function nn_utils.prepareNetworkForSave(node, nogc)
    node:clearState()
    if node.output ~= nil then
        node.output = nn_utils.zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = nn_utils.zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = nn_utils.zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                nn_utils.prepareNetworkForSave(child, true)
            end
        end
    end
    if nogc == nil or nogc == false then
        collectgarbage()
    end
end

function nn_utils.getNumberOfParameters(net)
    local nparams = 0
    local dModules = net:listModules()
    for i=1,#dModules do
        if dModules[i].weight ~= nil then
            nparams = nparams + dModules[i].weight:nElement()
        end
    end
    return nparams
end

-- Contains the pixels necessary to draw digits 0 to 9
CHAR_TENSORS = {}
CHAR_TENSORS[0] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {1, 1, 1}}):float()
CHAR_TENSORS[1] = torch.Tensor({{0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1}}):float()
CHAR_TENSORS[2] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1}}):float()
CHAR_TENSORS[3] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {0, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}}):float()
CHAR_TENSORS[4] = torch.Tensor({{1, 0, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {0, 0, 1},
                                {0, 0, 1}}):float()
CHAR_TENSORS[5] = torch.Tensor({{1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}}):float()
CHAR_TENSORS[6] = torch.Tensor({{1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1}}):float()
CHAR_TENSORS[7] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1}}):float()
CHAR_TENSORS[8] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1}}):float()
CHAR_TENSORS[9] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}}):float()

-- Converts a list of images to a grid of images that can be saved easily.
-- It will also place the epoch number at the bottom of the image.
-- At least parts of this function probably should have been a simple call
-- to image.toDisplayTensor().
-- @param images Tensor of image tensors
-- @param height Height of the grid
-- @param width Width of the grid
-- @param epoch The epoch number to draw at the bottom of the grid
-- @returns tensor
function nn_utils.imagesToGridTensor(images, height, width, epoch)
    local imgChannels = images:size(2)
    local imgHeightPx = IMG_DIMENSIONS[2]
    local imgWidthPx = IMG_DIMENSIONS[3]
    local heightPx = height * imgHeightPx + (1 + 5 + 1)
    local widthPx = width * imgWidthPx
    local grid = torch.zeros(imgChannels, heightPx, widthPx):float()

    -- add images to grid, one by one
    local yGridPos = 1
    local xGridPos = 1
    for i=1,math.min(images:size(1), height*width) do
        -- set pixels of image
        local yStart = 1 + ((yGridPos-1) * imgHeightPx)
        local yEnd = yStart + imgHeightPx - 1
        local xStart = 1 + ((xGridPos-1) * imgWidthPx)
        local xEnd = xStart + imgWidthPx - 1
        grid[{{1,imgChannels}, {yStart,yEnd}, {xStart,xEnd}}] = images[i]:float()

        -- move to next position in grid
        xGridPos = xGridPos + 1
        if xGridPos > width then
            xGridPos = 1
            yGridPos = yGridPos + 1
        end
    end

    -- add the epoch at the bottom of the image
    local epochStr = tostring(epoch)
    local pos = 1
    for i=epochStr:len(),1,-1 do
        local c = tonumber(epochStr:sub(i,i))
        for channel=1,imgChannels do
            local yStart = heightPx - 1 - 5 -- constant for all
            local yEnd = yStart + 5 - 1 -- constant for all
            local xStart = widthPx - 1 - pos*5 - pos
            local xEnd = xStart + 3 - 1

            grid[{{channel}, {yStart, yEnd}, {xStart, xEnd}}] = CHAR_TENSORS[c]
        end
        pos = pos + 1
    end

    return grid
end

-- Saves the list of image to the provided filepath (as a grid image).
-- @param filepath Save the grid image to that filepath
-- @param images List of image tensors
-- @param height Height of the grid
-- @param width Width of the grid
-- @param epoch The epoch number to draw at the bottom of the grid
-- @returns tensor
function nn_utils.saveImagesAsGrid(filepath, images, height, width, epoch)
    local grid = nn_utils.imagesToGridTensor(images, height, width, epoch)
    os.execute(string.format("mkdir -p %s", sys.dirname(filepath)))
    image.save(filepath, grid)
end

-- Deactivates CUDA mode on a network and returns the result.
-- Expects networks in CUDA mode to be a Sequential of the form
-- [1] Copy layer [2] Sequential [3] Copy layer
-- as created by activateCuda().
-- @param net The network to deactivate CUDA mode on.
-- @returns The CPU network
function nn_utils.deactivateCuda(net)
    local newNet = net:clone()
    newNet:float()
    if torch.type(newNet:get(1)) == 'nn.Copy' then
        return newNet:get(2)
    else
        return newNet
    end
end

-- Returns whether a Sequential contains any copy layers.
-- @param net The network to analyze.
-- @return boolean
function nn_utils.containsCopyLayers(net)
    local modules = net:listModules()
    for i=1,#modules do
        local t = torch.type(modules[i])
        if string.find(t, "Copy") ~= nil then
            return true
        end
    end
    return false
end

-- Activates CUDA mode on a network and returns the result.
-- This adds Copy layers at the start and end of the network.
-- Expects the default tensor to be FloatTensor.
-- @param net The network to activate CUDA mode on.
-- @returns The CUDA network
function nn_utils.activateCuda(net)
    --[[
    local newNet = net:clone()
    newNet:cuda()
    local tmp = nn.Sequential()
    tmp:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    tmp:add(newNet)
    tmp:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
    return tmp
    --]]
    local newNet = net:clone()

    -- does the network already contain any copy layers?
    local containsCopyLayers = nn_utils.containsCopyLayers(newNet)

    -- no copy layers in the network yet
    -- add them at the start and end
    if not containsCopyLayers then
        local tmp = nn.Sequential()
        tmp:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
        tmp:add(newNet)
        tmp:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
        newNet:cuda()
        newNet = tmp
    end

    --[[
    local firstCopyFound = false
    local lastCopyFound = false
    modules = newNet:listModules()
    for i=1,#modules do
        print("module "..i.." " .. torch.type(modules[i]))
        local t = torch.type(modules[i])
        if string.find(t, "Copy") ~= nil then
            if not firstCopyFound then
                firstCopyFound = true
                modules[i]:cuda()
                modules[i].intype = 'torch.FloatTensor'
                modules[i].outtype = 'torch.CudaTensor'
            else
                -- last copy found
                lastCopyFound = true
                modules[i]:float()
                modules[i].intype = 'torch.CudaTensor'
                modules[i].outtype = 'torch.FloatTensor'
            end
        elseif lastCopyFound then
            print("calling float() A")
            modules[i]:float()
        elseif firstCopyFound then
            print("calling cuda()")
            modules[i]:cuda()
        else
            print("calling float() B")
            modules[i]:float()
        end
    end
    --]]

    return newNet
end

-- Creates an average rating (0 to 1) for a list of images.
-- 1 is best.
-- @param images List of image tensors.
-- @returns float
function nn_utils.rateWithV(images)
    local imagesTensor
    local N
    if type(images) == 'table' then
        N = #images
        imagesTensor = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        for i=1,N do
            imagesTensor[i] = images[i]
        end
    else
        N = images:size(1)
        imagesTensor = images
    end

    local predictions = MODEL_V:forward(imagesTensor)
    local sm = 0
    for i=1,N do
        -- first neuron in V signals whether the image is fake (1=yes, 0=no)
        sm = sm + predictions[i][1]
    end

    local fakiness = sm / N

    -- higher values for better images
    return (1 - fakiness)
end

return nn_utils
