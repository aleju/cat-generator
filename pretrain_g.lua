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
    --batchSize     (default 16)
    --noplot                            Whether to not plot
    --window        (default 23)
    --seed          (default 1)
    --aws                               run in AWS mode
    --saveFreq      (default 5)        
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --colorSpace    (default "rgb")     rgb|yuv|hsl|y
    --scale         (default 32)
    --G_clamp       (default 5)
    --G_L1          (default 0)
    --G_L2          (default 0)
    --N_epoch       (default 10000)
    --noiseDim      (default 100)
]]

NORMALIZE = false
if OPT.colorSpace == "y" then
    OPT.grayscale = true
end

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
torch.setnumthreads(OPT.threads)

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
torch.setdefaulttensortype('torch.FloatTensor')


function main()
    ----------------------------------------------------------------------
    -- get/create dataset
    ----------------------------------------------------------------------
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
    
    -- Initialize G in autoencoder form
    -- G is a Sequential that contains (1) G Encoder and (2) G Decoder (both again Sequentials)
    G_AUTOENCODER = MODELS.create_G_autoencoder(IMG_DIMENSIONS, OPT.noiseDim)
    
    if OPT.gpu then
        G_AUTOENCODER = NN_UTILS.activateCuda(G_AUTOENCODER)
    end
    
    print("G autoencoder:")
    print(G_AUTOENCODER)
    print(string.format('Number of free parameters in G (total): %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER)))
    if OPT.gpu ~= false then
        print(string.format('... encoder: %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER:get(1))))
        print(string.format('... decoder: %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER:get(2))))
    else
        print(string.format('... encoder: %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER:get(2):get(1))))
        print(string.format('... decoder: %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER:get(2):get(2))))
    end
    
    -- Mean squared error criterion
    CRITERION = nn.MSECriterion()
    
    -- Get parameters and gradients
    PARAMETERS_G_AUTOENCODER, GRAD_PARAMETERS_G_AUTOENCODER = G_AUTOENCODER:getParameters()
    
    -- Initialize adam state
    OPTSTATE = {adam={}}
    
    if NORMALIZE then
        TRAIN_DATA = DATASET.loadRandomImages(10000)
        NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
    end
    
    -- training loop
    EPOCH = 1
    while true do
        print(string.format("<trainer> Epoch %d", EPOCH))
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        if NORMALIZE then
            TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        end
        
        epoch()
        
        if not OPT.noplot then
            visualizeProgress()
        end
    end
end

-- Train G (in autoencoder form) for one epoch
function epoch()
    local startTime = sys.clock()
    local batchIdx = 0
    local trained = 0
    -- minibatch loop
    -- keep training until we have reached the requested number of samples per epoch
    while trained < OPT.N_epoch do
        -- size of this batch, usually OPT.batchSize, may be smaller at the end 
        local thisBatchSize = math.min(OPT.batchSize, OPT.N_epoch - trained)
        local inputs = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        
        -- as G is in autoencoder form, input (x) and target (y) are both the same image(s)
        for i=1,thisBatchSize do
            inputs[i] = TRAIN_DATA[i]:clone()
            targets[i] = TRAIN_DATA[i]:clone()
        end
        
        -- evaluation function for G
        local fevalG = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G_AUTOENCODER then PARAMETERS_G_AUTOENCODER:copy(x) end
            GRAD_PARAMETERS_G_AUTOENCODER:zero()

            --  forward pass
            local outputs = G_AUTOENCODER:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            G_AUTOENCODER:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G_AUTOENCODER, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G_AUTOENCODER, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G_AUTOENCODER:add(torch.sign(PARAMETERS_G_AUTOENCODER):mul(OPT.G_L1) + PARAMETERS_G_AUTOENCODER:clone():mul(OPT.G_L2) )
            end

            -- Clamp G's gradients
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G_AUTOENCODER:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end
            
            return f,GRAD_PARAMETERS_G_AUTOENCODER
        end
        
        -- use Adam as optimizer
        optim.adam(fevalG, PARAMETERS_G_AUTOENCODER, OPTSTATE.adam)
        
        trained = trained + thisBatchSize
        batchIdx = batchIdx + 1
        
        xlua.progress(trained, OPT.N_epoch)
    end
    
    -- Epoch has finished (all batches done)

    -- Some outputs for this epoch
    local epochTime = sys.clock() - startTime
    print(string.format("<trainer> time required for this epoch = %d s", epochTime))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * epochTime/OPT.N_epoch))
    print(string.format("<trainer> last batch loss: %.4f", CRITERION.output))

    -- save the model    
    if EPOCH % OPT.saveFreq == 0 then
        -- filename is "g_pretrained_CHANNELSxHEIGHTxWIDTH_NOISEDIM.net"
        -- where NOISEDIM is equal to the size of layer between encoder and decoder (z)
        local filename = paths.concat(OPT.save, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        print(string.format("<trainer> saving network to %s", filename))
        
        -- Clone the autoencoder and deactivate cuda mode
        local G2 = G_AUTOENCODER:clone()
        G2:float()
        G2 = NN_UTILS.deactivateCuda(G2)
        
        -- :get(2) because we only want the decode part
        torch.save(filename, {G=G2:get(2), opt=OPT, EPOCH=EPOCH+1})
    end
    
    EPOCH = EPOCH + 1
end

-- Function to plot the current autoencoder training progress,
-- i.e. show training images and images after encode-decode
function visualizeProgress()
    -- deactivate dropout
    G_AUTOENCODER:evaluate()
    
    -- This global static array will be used to save the loss function values
    if PLOT_DATA == nil then PLOT_DATA = {} end
    
    -- Load some images
    -- we will only test here on potential training images
    local imagesReal = DATASET.loadRandomImages(100)
    if NORMALIZE then
        imagesReal.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    end
    
    -- Convert them to a tensor (instead of list of tensors),
    -- :forward() and display (DISP) want that
    local imagesRealTensor = torch.Tensor(imagesReal:size(), IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,imagesReal:size() do imagesRealTensor[i] = imagesReal[i] end
    
    -- encode-decode the images
    local imagesAfterG = G_AUTOENCODER:forward(imagesRealTensor)

    -- log the loss of the last encode-decode
    table.insert(PLOT_DATA, {EPOCH, CRITERION.output})
    
    -- display images, images after encode-decode, plot of loss function
    DISP.image(NN_UTILS.toRgb(imagesRealTensor, OPT.colorSpace), {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="Original images (before Autoencoder) (EPOCH " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(imagesAfterG, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Images after autoencoder G (EPOCH " .. EPOCH .. ")"})
    DISP.plot(PLOT_DATA, {win=OPT.window+2, labels={'epoch', 'G Loss'}, title='G Loss'})
    
    -- reactivate dropout
    G_AUTOENCODER:training()
end

main()
