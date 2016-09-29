--not ready out of box for CPU, remove all GPU lines
require 'nn'
require 'optim'
require 'image'
local tds = require 'tds'
local tnt = require 'torchnet'
require 'classificationlogger'
-- for fixing batchnorm duting fine tune
local inn = require 'inn'
local innutils = require 'inn.utils'
local utils = require 'utils'
local function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 (Torchnet) Imagenet Training script')
   cmd:text()
   cmd:text('Options:')
   cmd:text('---------- General options ----------------------------------')
   cmd:text()
   cmd:option('-data',        './data',      'Home of ImageNet dataset')
   cmd:option('-cache',       './imagenet',  'directory to log experiments')
   cmd:option('-netType',    'alexnetowtbn', 'Options: alexnetowtbn < for now')
   cmd:option('-GPU',         1,             'Default preferred GPU < only 1 if dpt')
   cmd:option('-nGPU',        1,             'Number of GPUs to use')
   cmd:option('-backend',     'cudnn',       'Options: cudnn | nn')
   cmd:option('-cudnn',       'fastest',     'Options: fastest | deterministic')
   cmd:option('-manualSeed',  2,             'Manually set RNG seed')
   cmd:text()
   cmd:text('---------- Data options ----------------------------------')
   cmd:text()
   cmd:option('-balanced',   'yes',         '(yes|no)')
   cmd:option('-nDonkeys',   8,    'number of data loading threads')
   cmd:option('-imageSize',  256,  'Smallest side of the resized image')
   cmd:option('-cropSize',   224,  'Height and Width of input layer')
   cmd:text()
   cmd:text('---------- Training options ----------------------------------')
   cmd:text()
   cmd:option('-nEpochs',    55,   'Number of total epochs to run')
   cmd:option('-batchSize',  128,  'mini-batch size (1 = pure stochastic)')
   cmd:text()
   cmd:text('---------- Optimization options ----------------------------------')
   cmd:text()
   cmd:option('-LR',         0.0,  'learning rate; 0 - default LR/WD recipe')
   cmd:option('-momentum',   0.9,  'momentum')
   cmd:option('-WD',         5e-4, 'weight decay')
   cmd:text()
   cmd:text('---------- Resume/Finetune options ----------------------------------')
   cmd:text()
   cmd:option('-retrain',    'none', 'model to retrain with')
   cmd:option('-optimState', 'none', 'optimState to reload from')
   cmd:option('-epoch',       0,     'epochs completed (for LR Policy)')
   cmd:option('-reset',      'no', '(yes|no) - reset last layer')
   cmd:option('-fixBN',      'no', '(yes|no) - fix BN Layers')
   cmd:option('-lr_factor',  0.1, 'lr factor for pretrained network layers')
   cmd:text()

   local opt = cmd:parse(arg or {})
   -- add commandline specified options
   opt.save = paths.concat(opt.cache, cmd:string(opt.netType, opt,
      {netType=true, retrain=true, optimState=true, cache=true, data=true}))
   -- add date/time
   opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
   opt.reset = opt.reset == 'yes'
   opt.fixBN = opt.fixBN == 'yes'
   opt.balanced = opt.balanced == 'yes'
   local function ls(x) return sys.split(sys.ls(x:gsub(" ","\\ ")),'\n') end
   opt.nClasses = #ls(paths.concat(opt.data,'train'))
   return opt
end

local opt = parse(arg)
print(opt)

if opt.backend == 'cudnn' then
   require 'cunn'
   require 'cudnn'
   cutorch.setDevice(opt.GPU)
   cutorch.manualSeedAll(opt.manualSeed)
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end
end
------------------
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
paths.mkdir(opt.save)
pretty.dump(opt,paths.concat(opt.save,'opts.txt'))

local config = {
   LR = opt.LR,
   learningRate = opt.LR,
   learningRateDecay = 0.0,
   dampening = 0.0,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
}
local optimState = opt.optimState ~= 'none' and torch.load(opt.optimState) or nil

---------------
local model
if opt.retrain ~= 'none' then
   model = torch.load(opt.retrain)
   if opt.reset then
      model.modules[2].modules[7] = nn.Sequential()
         :add(nn.GradientReversal(-1*opt.lr_factor))
         :add(nn.Linear(4096,4096))
      model.modules[2].modules[10] = nn.Linear(4096,opt.nClasses)
   end
   -- Fix batchnorm layers for fine tuning
   if opt.fixBN then
      innutils.BNtoFixed(model,false)
   end
else
   print('=> Creating model from file: ' .. opt.netType)
   model = paths.dofile(opt.netType .. '.lua')(opt)
end

local criterion = nn.ClassNLLCriterion()

if opt.backend == 'cudnn' then
   model:cuda()
   criterion:cuda()
   cudnn.convert(model, cudnn)
   -- only for alexnet, we make only 1 parallel
   model.modules[1] = utils.makeDataParallelTable(model.modules[1],opt.nGPU)
elseif opt.backend ~= 'nn' then
   error'Unsupported backend'
end

-- local sample_input = torch.randn(8,3,opt.cropSize,opt.cropSize):cuda()
-- local optnet = require 'optnet'
-- optnet.optimizeMemory(model, sample_input, {inplace = false, mode = 'training'})
print(model)
---------------
-- returns a tds structure with all information about the dataset
-- cannot be combined with iterator because that gives problems with random
local provider = utils.getImagenetProvider(
   opt.data,
   paths.concat(opt.cache,'dataCache.t7'),
   paths.concat(opt.cache,'meanStdCache.t7')
)

local function getIterator(mode)
   local iterator = tnt.ParallelDatasetIterator{
      nthread = opt.nDonkeys,
      init = function(nthread)
         require 'torchnet'
         require 'randomdataset'
         require 'image'
         -- from fb.resnet.torch
         transforms = require 'transforms'
         --seed has to be kept different for all threads as random is called
         torch.manualSeed(opt.manualSeed+nthread+1) 
      end,
      closure = function()
         local dataset = provider[mode]
         local classes = provider.classes
         local classlist= {}
         for l= 1,#dataset do
            classlist[#classlist+1] = tnt.ListDataset{
               list = dataset[l],
               load = function(im)
                  return {
                     input = image.load(paths.concat(opt.data,mode,classes[l],im),3,'float'),
                     target = torch.LongTensor{l}
                  }
               end,
            }
         end
         if mode == 'train' then
            local avg_size = math.floor(provider['trainSize']/#provider.classes)
            for k=1,#classlist do
               classlist[k] = opt.balanced and classlist[k]:random(avg_size) or classlist[k]:random()
            end
         end
         local concat =  tnt.ConcatDataset{datasets = classlist}
         -- keep ordering same in val, in train acts as class selector
         if mode == 'train' then concat = concat:random() end

         return concat:transform{
            input =
               mode == 'train' and
                  tnt.transform.compose{
                     transforms.Scale(opt.imageSize),
                     transforms.RandomCrop(opt.cropSize),
                     transforms.ColorNormalize(provider.meanstd),
                     transforms.HorizontalFlip(0.5),
                  }
              or
                  tnt.transform.compose{
                     transforms.Scale(opt.imageSize),
                     transforms.CenterCrop(opt.cropSize),
                     transforms.ColorNormalize(provider.meanstd),
                  }
               }:batch(opt.batchSize,'skip-last')
      end,
   }
   return iterator
end
------------
local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local logger = ClassificationLogger(opt.save, opt.nClasses)
------------
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
   state.epoch = opt.epoch or 0
end

engine.hooks.onStartEpoch = function(state)
   local epoch = state.epoch + 1
   state.config = utils.ImageNetRegime(state.config,epoch)
   timers.epochTimer:reset()
end

local inputs = torch.Tensor():cuda()
local targets = torch.Tensor():cuda()
engine.hooks.onSample = function(state)
   cutorch.synchronize()
   timers.dataTimer:stop()
   inputs:resize(state.sample.input:size()):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input = inputs
   state.sample.target = targets:squeeze()
end

engine.hooks.onForwardCriterion = function(state)
   if state.training then
      local temp_acc = logger:train(state.criterion.output,state.network.output,
         state.sample.target)
      print(('Epoch:%d [%d]   [Data/BatchTime %.3f/%.3f]   LR %.0e   Err %.4f Top1 %.2f'):format(
         state.epoch+1, state.t, timers.dataTimer:time().real, 
         timers.batchTimer:time().real, state.config.learningRate,
         state.criterion.output, temp_acc))
      timers.batchTimer:reset() -- cycle can start anywhere
   else
      logger:val(state.criterion.output,state.network.output,
         state.sample.target)
   end
end

engine.hooks.onUpdate = function(state)
   cutorch.synchronize()
   timers.dataTimer:reset()
   timers.dataTimer:resume()
end

engine.hooks.onEndEpoch = function(state)
   print("Total Epoch time (Train):",timers.epochTimer:time().real)
   print('Testing')
   engine:test{
      network = model,
      iterator = getIterator('val'),
      criterion = criterion,
   }
   logger:endEpoch(state.epoch)
   state.t = 0
   torch.save(paths.concat(opt.save,'optim_' .. state.epoch ..'.t7'),
      state.optim)
   local clone = utils.checkpoint(state.network)
   clone.classes = provider.classes
   clone.meanstd = provider.meanstd
   if opt.backend == 'cudnn' then cudnn.convert(clone,nn) end
   torch.save(paths.concat(opt.save,'model_' .. state.epoch ..'.t7'),
      clone)
end

engine:train{
   network = model,
   iterator = getIterator('train'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = config,
   optimState = optimState,
   maxepoch = opt.nEpochs,
}
