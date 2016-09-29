local utils = {}
local tnt = require 'torchnet'
local tds = require 'tds'

function utils.makeDataParallelTable(model, nGPU) -- doesnt work with setDevice
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end

function utils.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

function utils.DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k,v in pairs(tbl) do
      -- will skip all DPTs. it also causes stack overflow, idk why
      if torch.typename(v) == 'nn.DataParallelTable' then
         v = v:get(1)
      end
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

utils.deepCopy = deepCopy

function utils.checkpoint(net)
   return deepCopy(net):float():clearState()
end

function utils.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = utils.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resize(t2:size()):copy(t2)
   elseif torch.type(t2) == 'number' then
      t1 = t2
   else
      error("expecting nested tensors or tables. Got "..
      torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function utils.recursiveCast(dst, src, type)
   if #dst == 0 then
      tnt.utils.table.copy(dst, nn.utils.recursiveType(src, type))
   end
   utils.recursiveCopy(dst, src)
   return dst
end

function utils.ImageNetRegime(config, epoch,regimes)
   if not config.LR then error('ImageNetRegime - Necessary to have LR field.') end
    if config.LR ~= 0.0 then -- if manually specified
        return config
    end
    local regimes = regimes or {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
           config.learningRate=row[3]
           config.weightDecay=row[4]
           return config, epoch == row[1]
        end
    end
end

function utils.getImagenetProvider(source,dataCache,meanStdCache)
   local ret = tds.Hash()
   assert(paths.dirp(source),'Source folder missing')
   assert(paths.dirp(paths.concat(source,'train')) and paths.dirp(paths.concat(source,'val')),
      'Source folder should have train and val folders')
   if dataCache and paths.filep(dataCache) then
      local data = torch.load(dataCache)
      ret.train = data.train
      ret.val = data.val
      ret.trainSize = data.trainSize
      ret.valSize = data.valSize
      ret.classes = data.classes
   else
      print('Preparing data')
      local function ls(dir) return sys.split(sys.ls(dir:gsub(" ","\\ ")),'\n') end
      local classes = tds.Hash(ls(paths.concat(source,'train')))
      ntr,nva = 0,0
      train,val = tds.Hash(),tds.Hash()
      for k=1,#classes do
         local v = classes[k]
         io.write('.')
         local trainfiles = tds.Hash(ls(paths.concat(source,'train',v)))
         local valfiles = tds.Hash(ls(paths.concat(source,'val',v)))
         train[#train+1] = trainfiles
         val[#val+1] = valfiles
         ntr = ntr + #trainfiles
         nva = nva + #valfiles
      end
      ret.train = train
      ret.val = val
      ret.trainSize = ntr
      ret.valSize = nva
      ret.classes = classes
      if dataCache then torch.save(dataCache,ret) end
   end
   if meanStdCache then
      if paths.filep(meanStdCache) then
         ret.meanstd = torch.load(meanStdCache)
         print(ret.meanstd)
      else
         local nSamples = 10000
         print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
         local avg_size = math.floor(ret.trainSize/#ret.classes)
         local classlist= {}
         for l=1,#ret.train do
            local list = tnt.ListDataset{
               list = ret.train[l],
               load = function(im)
                  return image.load(paths.concat(source,'train',ret.classes[l],im),3,'float')
               end,
            }
            classlist[#classlist+1] = list:shuffle(avg_size,true)
         end
         local iter =  tnt.ConcatDataset{datasets = classlist}:shuffle(nSamples,true):iterator()
         local tm = torch.Timer()
         local meanEstimate = {0,0,0}
         local stdEstimate = {0,0,0}
         for img in iter() do
            for j=1,3 do
               meanEstimate[j] = meanEstimate[j] + img[j]:mean()
               stdEstimate[j] = stdEstimate[j] + img[j]:std()
            end
         end
         for j=1,3 do
            meanEstimate[j] = meanEstimate[j] / nSamples
            stdEstimate[j] = stdEstimate[j] / nSamples
         end
         ret.meanstd = tds.Hash({mean = meanEstimate,std = stdEstimate})
         print(ret.meanstd)
         print('Time to estimate:', tm:time().real)
         torch.save(meanStdCache,ret.meanstd)
      end
   end
   collectgarbage()
   collectgarbage()
   return ret
end

return utils
