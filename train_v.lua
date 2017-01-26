require 'torch'
require 'image'
require 'paths'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'optim'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'

OPT = lapp[[
    --save          (default "logs")
    --batchSize     (default 32)
    --noplot                            Whether to not plot
    --window        (default 13)
    --seed          (default 1)
    --aws                               run in AWS mode
    --saveFreq      (default 10)
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --colorSpace    (default "rgb")     rgb|yuv|hsl|y
    --scale         (default 32)
    --V_clamp       (default 5)
    --V_L1          (default 0)
    --V_L2          (default 0.01)
    --N_epoch       (default 1000)
]]

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
torch.setnumthreads(OPT.threads)

Y_FAKE = 0
Y_REAL = 1
CLASSES = {"0", "1"}

if OPT.colorSpace == "y" then
    OPT.grayscale = true
end

if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.scale, OPT.scale}
else
    IMG_DIMENSIONS = {3, OPT.scale, OPT.scale}
end

INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]

-- run on gpu if chosen
if OPT.gpu then
    print("<trainer> starting gpu support...")
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
else
    require 'nn'
end
require 'dpnn'
require 'LeakyReLU'
torch.setdefaulttensortype('torch.FloatTensor')


-- Main function, load data, create network, start training.
function main()
    ----------------------------------------------------------------------
    -- get/create dataset
    ----------------------------------------------------------------------
    --DATASET.nbChannels = IMG_DIMENSIONS[1]
    DATASET.colorSpace = OPT.colorSpace
    DATASET.setFileExtension("jpg")
    DATASET.setHeight(OPT.scale)
    DATASET.setWidth(OPT.scale)

    if OPT.aws then
        DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
    else
        DATASET.setDirs({"dataset/out_aug_64x64"})
    end
    ----------------------------------------------------------------------

    V = MODELS.create_V(IMG_DIMENSIONS)

    if OPT.gpu then V = NN_UTILS.activateCuda(V) end

    print("network V:")
    print(V)

    CRITERION = nn.BCECriterion()
    PARAMETERS_V, GRAD_PARAMETERS_V = V:getParameters()
    CONFUSION = optim.ConfusionMatrix(CLASSES)
    OPTSTATE = {adam={}}
    -- normalization is currently deactivated
    --TRAIN_DATA = DATASET.loadRandomImages(10000)
    --NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()

    EPOCH = 1
    while true do
        print(string.format("<trainer> Epoch %d", EPOCH))
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        --TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        epoch(V)
        if not OPT.noplot then
            visualizeProgress()
        end
    end
end

-- Run one epoch of training.
function epoch()
    local startTime = sys.clock()
    local batchIdx = 0
    local trained = 0
    while trained < OPT.N_epoch do
        local thisBatchSize = math.min(OPT.batchSize, OPT.N_epoch - trained)
        local inputs = torch.zeros(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):float()
        local targets = torch.zeros(thisBatchSize, 2):float()

        local fevalV = function(x)
            collectgarbage()
            if x ~= PARAMETERS_V then PARAMETERS_V:copy(x) end
            GRAD_PARAMETERS_V:zero()

            --  forward pass
            local outputs = V:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass
            local df_do = CRITERION:backward(outputs, targets)
            V:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.V_L1 ~= 0 or OPT.V_L2 ~= 0 then
                -- Loss:
                f = f + OPT.V_L1 * torch.norm(PARAMETERS_V, 1)
                f = f + OPT.V_L2 * torch.norm(PARAMETERS_V, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_V:add(torch.sign(PARAMETERS_V):mul(OPT.V_L1) + PARAMETERS_V:clone():mul(OPT.V_L2) )
            end

            -- update confusion (add 1 since targets are binary)
            for i = 1,thisBatchSize do
                local predictedClass
                local realClass
                if outputs[i][1] > 0.5 then predictedClass = 0 else predictedClass = 1 end
                if targets[i][1] == 1 then realClass = 0 else realClass = 1 end
                CONFUSION:add(predictedClass+1, realClass+1)
            end

            -- Clamp V's gradients
            if OPT.V_clamp ~= 0 then
                GRAD_PARAMETERS_V:clamp((-1)*OPT.V_clamp, OPT.V_clamp)
            end

            return f,GRAD_PARAMETERS_V
        end

        --------------------------------------
        -- Collect examples for batch
        --------------------------------------
        -- Real data
        local exampleIdx = 1
        local realDataSize = thisBatchSize / 2
        for i=1,thisBatchSize/2 do
            local randomIdx = math.random(TRAIN_DATA:size())
            inputs[exampleIdx] = TRAIN_DATA[randomIdx]:clone()
            targets[exampleIdx][Y_REAL+1] = 1
            targets[exampleIdx][Y_FAKE+1] = 0
            exampleIdx = exampleIdx + 1
        end

        -- Fake data
        local images = imageListToTensor(createSyntheticImages(thisBatchSize/2, V))
        --NN_UTILS.normalize(images, NORMALIZE_MEAN, NORMALIZE_STD)
        for i = 1, realDataSize do
            inputs[exampleIdx] = images[i]:clone()
            targets[exampleIdx][Y_REAL+1] = 0
            targets[exampleIdx][Y_FAKE+1] = 1
            exampleIdx = exampleIdx + 1
        end

        optim.adam(fevalV, PARAMETERS_V, OPTSTATE.adam)


        trained = trained + thisBatchSize
        batchIdx = batchIdx + 1

        xlua.progress(trained, OPT.N_epoch)
    end

    local epochTime = sys.clock() - startTime
    print(string.format("<trainer> time required for this epoch = %d s", epochTime))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * epochTime/OPT.N_epoch))
    print("Confusion of V:")
    print(CONFUSION)
    CONFUSION:zero()

    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, string.format('v_%dx%dx%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        print(string.format("<trainer> saving network to %s", filename))

        NN_UTILS.prepareNetworkForSave(V)
        torch.save(filename, {V=V, opt=OPT, EPOCH=EPOCH+1})
    end

    EPOCH = EPOCH + 1
end

-- Convert a list (table) of images to a Tensor.
-- @param imageList A list/table of images (tensors).
-- @returns A tensor of shape (N, channels, height, width)
function imageListToTensor(imageList)
    local tens = torch.zeros(#imageList, imageList[1]:size(1), imageList[1]:size(2), imageList[1]:size(3)):float()
    for i=1,#imageList do
        tens[i] = imageList[i]
    end
    return tens
end

-- Create plots showing the current training progress.
function visualizeProgress()
    -- deactivate dropout
    V:evaluate()

    -- gather 50 real images and 50 fake (synthetic) images
    local imagesReal = DATASET.loadRandomImages(50)
    local imagesFake = imageListToTensor(createSyntheticImages(50))
    --imagesReal.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    --NN_UTILS.normalize(imagesFake, NORMALIZE_MEAN, NORMALIZE_STD)
    local both = torch.zeros(100, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):float()
    for i=1,imagesReal:size(1) do
        both[i] = imagesReal[i]
    end
    for i=1,imagesFake:size(1) do
        both[imagesReal:size(1) + i] = imagesFake[i]
    end

    -- let V judge the 50 real and 50 fake images
    -- each single prediction is an array [p(fake), p(real)]
    local predictions = V:forward(both)

    -- separate the images in good/bad one based on what
    -- V thinks about them, i.e. based on p(fake)
    local goodImages = {}
    local badImages = {}
    for i=1,predictions:size(1) do
        -- these if statements help spotting problems in the generation
        -- of the synthetic images
        if torch.any(both[i]:gt(1.0)) then
            print("[WARNING] bad values in image")
            print(both[i][both[i]:gt(1.0)])
            print("image i=", i, " is ge1")
        end
        if torch.any(both[i]:lt(0.0)) then
            print("[WARNING] bad values in image")
            print(both[i][both[i]:lt(0.0)])
            print("image i=", i, " is lt0")
        end

        -- add image i to good or bad images, depending on
        -- what V thinks about it
        if predictions[i][1] < 0.5 then
            goodImages[#goodImages+1] = both[i]
        else
            badImages[#badImages+1] = both[i]
        end
    end

    -- show the gathered good/bad images via display
    if #goodImages > 0 then
        DISP.image(NN_UTILS.toRgb(goodImages, OPT.colorSpace), {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="V: rated as real images (Epoch " .. EPOCH .. ")"})
    end
    if #badImages > 0 then
        DISP.image(NN_UTILS.toRgb(badImages, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="V: rated as fake images (EPOCH " .. EPOCH .. ")"})
    end

    -- reactivate dropout
    V:training()
end

-- Creates N synthetic (fake) images.
-- The sizes match the one set in OPT.scale.
-- @param N Number of images to create.
-- @param allowSubcalls Whether to allow mixing the created images with
--                      other synthetic images created via more calls
--                      to this method.
-- @returns List of torch.Tensor(channels, height, width)
function createSyntheticImages(N, allowSubcalls)
    if allowSubcalls == nil then allowSubcalls = true end
    local images

    local p = math.random()
    if p < 1/4 then
        images = createSyntheticImagesMix(N)
    elseif p >= 1/4 and p < 2/4 then
        images = createSyntheticImagesWarp(N)
    elseif p >= 2/4 and p < 3/4 then
        images = createSyntheticImagesStamp(N)
    else
        images = createSyntheticImagesRandom(N)
    end

    -- Mix with deeper calls to this method
    if allowSubcalls and math.random() < 0.33 then
        local otherImages = createSyntheticImages(N, false)
        images = mixImageLists(images, otherImages)
    end

    return images
end

-- Mixes two images according to an overlay/mask.
-- The overlay must be an image (2d tensor).
-- Images and overlay must have the same sizes in y and x.
-- Where values of the overlay are close to one, mostly image1 will be used.
-- @param img1 Tensor (channel, height, width) of the first image.
-- @param img2 Tensor (channel, height, width) of the second image.
-- @param overlay Tensor (height, width) with values between 0 and 1
--                or nil (to autogenerate a random overlay).
-- @returns Tensor (channel, height, width)
function mixImages(img1, img2, overlay)
    local img = torch.zeros(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):float()

    if overlay == nil then
        if math.random() < 0.5 then
            overlay = getGaussianOverlay()
        else
            overlay = createPixelwiseOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        end
    end

    overlay = torch.repeatTensor(overlay, IMG_DIMENSIONS[1], 1, 1)
    --    overlay * img1     + (1 - overlay) * img2
    img = overlay:clone():cmul(img1) + overlay:clone():mul(-1):add(1):cmul(img2)
    img:div(torch.max(img))

    return img
end

-- Applies mixImages to the pairs in two image lists.
-- @param images1 List of images.
-- @param images2 List of images.
-- @returns List of mixed images
function mixImageLists(images1, images2)
    local images = {}
    local overlay
    local p = math.random()
    if p < 0.5 then
        overlay = getGaussianOverlay()
    else
        overlay = createPixelwiseOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    end

    for i=1,#images1 do
        images[i] = mixImages(images1[i]:clone(), images2[i]:clone(), overlay)
    end

    return images
end

-- Creates N new images by mixing random images
-- from the training set.
-- @param N Number of images.
-- @returns List of Tensors (channel, height, width)
function createSyntheticImagesMix(N)
    local images = {}
    local img1 = {}
    local img2 = {}

    for i=1,N do
        table.insert(img1, TRAIN_DATA[math.random(TRAIN_DATA:size())])
        table.insert(img2, TRAIN_DATA[math.random(TRAIN_DATA:size())])
    end

    return mixImageLists(img1, img2)
end

-- Creates N new images by randomly copy-pasting areas within
-- random images from the training set
-- @param N Number of images.
-- @returns List of Tensors (channel, height, width)
function createSyntheticImagesStamp(N)
    local images = {}

    local nbChannels = TRAIN_DATA[1]:size(1)
    local maxY = TRAIN_DATA[1]:size(2)
    local maxX = TRAIN_DATA[1]:size(3)

    --local overlay = createGaussianOverlay(maxY, maxX, 1000+math.random(1000))
    local overlay = getGaussianOverlay()

    for i=1,N do
        local p = math.random()
        local img1 = TRAIN_DATA[math.random(TRAIN_DATA:size())]
        local img = torch.zeros(nbChannels, maxY, maxX):float()
        local direction = {math.random(10), math.random(10)}

        for y=1,maxY do
            for x=1,maxX do
                local coords = withinImageCoords(y + direction[1], x + direction[2], img:size(2), img:size(3))

                for c=1,nbChannels do
                    local usualVal = img1[c][y][x]
                    local sourceVal = img1[c][coords[1]][coords[2]]
                    --print("overlay:", overlay[y][x], "usualval:", usualVal, "sourceval:", sourceVal, "result:", (1 - overlay[y][x]) * usualVal + overlay[y][x] * sourceVal)
                    img[c][y][x] = (1 - overlay[y][x]) * usualVal + overlay[y][x] * sourceVal
                end
            end
        end
        img:div(torch.max(img))

        table.insert(images, img)
    end

    return images
end

-- Makes sure that given (y, x) coordinates are between 1 and maxY/maxX.
-- @param y Y-Coordinate
-- @param x X-Coordinate
-- @param maxY Max value for Y-Coordinate
-- @param maxX Max value for X-Coordinate
-- @returns List {y, x}
function withinImageCoords(y, x, maxY, maxX)
    y = y % maxY
    if y < 1 then
        y = y * (-1)
        y = maxY - y
    end

    x = x % maxX
    if x < 1 then
        x = x * (-1)
        x = maxX - x
    end

    return {y, x}
end

-- Creates N new images by randomly moving (warping) areas in random
-- images from the training set.
-- @param N Number of images.
-- @returns List of (channel, height, width)
function createSyntheticImagesWarp(N)
    local images = {}

    --local overlay1 = createGaussianOverlay(OPT.scale, OPT.scale)
    --local overlay2 = createGaussianOverlay(OPT.scale, OPT.scale)
    local overlay1 = getGaussianOverlay()
    local overlay2 = getGaussianOverlay()
    overlay1:mul(2.0)
    overlay1:add(-1.0)
    overlay2:mul(2.0)
    overlay2:add(-1.0)

    for i=1,N do
        local img1 = TRAIN_DATA[math.random(TRAIN_DATA:size())]:clone()
        local flow = torch.zeros(2, img1:size(2), img1:size(3)):float()

        local direction = {1, 0}
        local length = 1 + math.random(4.0)
        --local length = 3.0

        for y=1,img1:size(2) do
            for x=1,img1:size(3) do
                flow[1][y][x] = overlay1[y][x] * length
                flow[2][y][x] = overlay2[y][x] * length
            end
        end

        local img = image.warp(img1, flow)
        img:div(torch.max(img))

        table.insert(images, img)
    end

    return images
end

-- Creates N new images by randomly mixing gaussian overlays/masks in
-- different color channels.
-- @param N Number of images.
-- @returns List of tensors (channel, height, width)
function createSyntheticImagesRandom(N)
    local images = {}
    --local overlay1 = createGaussianOverlay(OPT.scale, OPT.scale, 2000, 10)
    --local overlay2 = createGaussianOverlay(OPT.scale, OPT.scale, 2000, 10)
    --local overlay3 = createGaussianOverlay(OPT.scale, OPT.scale, 10000, 4)
    local overlay1 = getGaussianOverlay(10)
    local overlay2 = getGaussianOverlay(10)

    for i=1,N do
        local overlay3 = getGaussianOverlay(4)

        local img = torch.zeros(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):float()
        --local offsetY = math.random(IMG_DIMENSIONS[2])
        --local offsetX = math.random(IMG_DIMENSIONS[3])
        local offsetY = math.random(10) - 5
        local offsetX = math.random(10) - 5
        local baseVal = {math.random(), math.random(), math.random()}

        --print(baseVal[0], baseVal[1], baseVal[2])

        for y=1,IMG_DIMENSIONS[2] do
            for x=1,IMG_DIMENSIONS[3] do
                for c=1,IMG_DIMENSIONS[1] do
                    local coords = withinImageCoords(y + c*offsetY, x + c*offsetX, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
                    img[c][y][x] = baseVal[c] + (overlay1[y][x] * overlay2[coords[1]][coords[2]]) - overlay3[coords[1]][coords[2]]
                end
            end
        end

        --image.display(img)
        --io.read()

        img:add(math.abs(torch.min(img)))
        img:div(torch.max(img))
        table.insert(images, img)
    end

    return images
end

-- Creates a new gaussian overlay/mask from a set of cached ones.
-- @param blurSize Defined how blurry the "clouds" in the mask are.
-- @returns Tensor (height, width)
function getGaussianOverlay(blurSize)
    if blurSize == nil then blurSize = 4 end

    if OVERLAYS == nil then
        OVERLAYS = {}
        for i=1,1000 do
            OVERLAYS[i] = createGaussianOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], 10000, 0)
        end
    end

    local overlay1 = OVERLAYS[math.random(#OVERLAYS)]:clone()
    local overlay2 = OVERLAYS[math.random(#OVERLAYS)]:clone()
    local overlay3 = OVERLAYS[math.random(#OVERLAYS)]:clone()
    local overlay4 = OVERLAYS[math.random(#OVERLAYS)]:clone()
    local overlayResult = overlay1:mul(2) - overlay2
    overlayResult = torch.clamp(overlayResult, 0.0, 1.0)
    overlayResult = overlayResult + overlay3:cmul(overlay4):mul(2)
    overlayResult = torch.clamp(overlayResult, 0.0, 1.0)

    if blurSize > 0 then
        overlayResult = image.convolve(overlayResult, image.gaussian(blurSize):float(), "same")
        overlayResult:div(torch.max(overlayResult))
    end

    --image.display(overlayResult)
    --io.read()

    return overlayResult
end

-- Creates a new gaussian overlay/mask.
-- It performs a random walk on the image canvas, leaving white dots behind.
-- It tends to walk around the same points and rarely jumps away to other
-- places, creating clusters of points.
-- The algorithm is pretty inefficient/slow.
-- @param ySize Height of the mask.
-- @param xSize Width of the mask.
-- @param N_points How many steps of random walk to perform.
-- @param blurSize How much to blur the result before returning it.
-- @returns Tensor (height, width)
function createGaussianOverlay(ySize, xSize, N_points, blurSize)
    N_points = N_points or 1000
    blurSize = blurSize or 6
    local minY = 1
    local maxY = ySize
    local minX = 1
    local maxX = xSize

    local overlay = torch.zeros(maxY, maxX):float()

    local directions = {
        {-1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
        {1, 0},
        {1, -1},
        {0, -1},
        {-1, -1}
    }

    local currentY = math.random(ySize)
    local currentX = math.random(xSize)
    local lastY = math.random(ySize)
    local lastX = math.random(xSize)

    for i=1,N_points do
        --print(i)
        local p = math.random()
        if p < 0.02 then
            lastY = currentY
            lastX = currentX
            currentY = math.random(maxY)
            currentX = math.random(maxX)
        elseif math.random() < 0.10 then
            currentY = lastY
            currentX = lastX
        else
            lastY = currentY
            lastX = currentX

            local found = false
            while not found do
                local direction = directions[math.random(#directions)]
                currentY = lastY + direction[1]
                currentX = lastX + direction[2]

                if (currentY >= minY and currentY <= maxY) and (currentX >= minX and currentX <= maxX) then
                    found = true
                end
            end
        end

        --print(currentY, currentX, overlay:size())
        overlay[currentY][currentX] = overlay[currentY][currentX] + 1
    end

    overlay:div(torch.max(overlay))
    if blurSize > 0 then
        overlay = image.convolve(overlay, image.gaussian(blurSize), "same")
        overlay:div(torch.max(overlay))
    end

    return overlay
end

-- Creates an overlay/mask by going through the image line by line
-- and setting pixels to random values between 0 and 1.
-- Pixels that are next to each other tend to get more similar values.
-- @param ySize Height of the mask.
-- @param xSize Width of the mask.
-- @returns Tensor (height, width)
function createPixelwiseOverlay(ySize, xSize)
    local overlay = torch.zeros(ySize, xSize):float()

    local p = math.random()
    local pChange = math.random() / 10

    for y=1,ySize do
        for x=1,xSize do
            if math.random() > p then
                overlay[y][x] = math.min(2*math.random(), 1)
            else
                overlay[y][x] = 0
            end

            if math.random() > 0.5 then
                p = math.max(p - pChange, 0)
            else
                p = math.min(p + pChange, 1.0)
            end
        end
    end

    return overlay
end

-- Makes sure that x is min_x <= x <= max_x.
-- @param min_x Min value for x.
-- @param max_x Max value for x.
-- @returns x within given bounds
function minmax(min_x, x, max_x)
    if x < min_x then
        return min_x
    elseif x > max_x then
        return max_x
    else
        return x
    end
end

main()
