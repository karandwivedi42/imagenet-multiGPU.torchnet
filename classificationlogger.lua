require 'optim'
local tnt = require 'torchnet'

local logger = torch.class('ClassificationLogger')
--[[

General purpose classification logger
has 4 functions: init, train, val and endEpoch.
See imagenet-multiGPU for ideal usage.
]]
function logger:__init(path,nClasses,prefix)
   self.__save = path
   self.__prefix = prefix or ''
   self.__trainmeters = {
      loss = tnt.AverageValueMeter(),
      acc_temp = tnt.ClassErrorMeter{topk = {1},accuracy=true},
      acc = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   }

   self.__valmeters = {
      conf = tnt.ConfusionMeter{k = nClasses, normalized = true},
      loss = tnt.AverageValueMeter(),
      acc = tnt.ClassErrorMeter{topk = {1},accuracy=true},
      ap = tnt.APMeter(),
   }

   self.__logs = {
      accuracy = optim.Logger(paths.concat(self.__save,self.__prefix..'acc.log')),
      loss = optim.Logger(paths.concat(self.__save,self.__prefix..'loss.log')),
      map = optim.Logger(paths.concat(self.__save,self.__prefix..'map.log')),
      full_train = optim.Logger(paths.concat(self.__save,self.__prefix..'full_train.log')),
   }

   self.__logs.accuracy:setNames{'Train Accuracy', 'Test Accuracy'}
   self.__logs.loss:setNames{'Train Loss', 'Test Loss'}
   self.__logs.map:setNames{'Test mAP'}
   self.__logs.full_train:setNames{'Train Loss'}

   self.__logs.accuracy.showPlot = false
   self.__logs.loss.showPlot = false
   self.__logs.map.showPlot = false
   self.__logs.full_train.showPlot = false
end

function logger:val(a,b,c)
   self.__valmeters.conf:add(b,c)
   self.__valmeters.acc:add(b,c)
   self.__valmeters.loss:add(a)
   local tar = torch.ByteTensor(#b):fill(0)
   for k=1,c:size(1) do
      tar[k][c[k]]=1
   end
   self.__valmeters.ap:add(b,tar)
end

function logger:train(a,b,c)
   self.__logs.full_train:add{a}
   self.__trainmeters.loss:add(a)
   self.__trainmeters.acc:add(b,c)
   self.__trainmeters.acc_temp:reset()
   self.__trainmeters.acc_temp:add(b,c)
   return self.__trainmeters.acc_temp:value()[1]
end

function logger:endEpoch(n)
   self.__logs.accuracy:add{self.__trainmeters.acc:value()[1],self.__valmeters.acc:value()[1]}
   self.__logs.map:add{self.__valmeters.ap:value():mean()}
   local temp1, temp2 = self.__trainmeters.loss:value()
   local temp3, temp4 = self.__valmeters.loss:value()
   self.__logs.loss:add{temp1, temp3}
   if self.__prefix ~= '' then print(self.__prefix .. ' :') end
   print('Train Loss:', temp1, 'Val Loss:', temp3)
   print('Train Acc:',self.__trainmeters.acc:value()[1],
      "Val Acc:", self.__valmeters.acc:value()[1])
   print("Val mAP:",self.__valmeters.ap:value():mean())

   self.__logs.accuracy:style{'+-','+-'}
   self.__logs.loss:style{'+-','+-'}
   self.__logs.map:style{'+-'}
   self.__logs.full_train:style{'-'}
   self.__logs.accuracy:plot()
   self.__logs.loss:plot()
   self.__logs.map:plot()
   self.__logs.full_train:plot()

   image.save(paths.concat(self.__save,self.__prefix..'confusion_' .. n ..'.jpg'),
      image.scale(self.__valmeters.conf:value():float(),1000,1000,'simple'))

   self.__trainmeters.loss:reset()
   self.__trainmeters.acc:reset()
   self.__valmeters.conf:reset()
   self.__valmeters.loss:reset()
   self.__valmeters.acc:reset()
   self.__valmeters.ap:reset()
end
