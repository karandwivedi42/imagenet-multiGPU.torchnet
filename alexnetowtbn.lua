local function createModel(opt)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(64))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(192))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(384))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.BatchNormalization(4096))
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.BatchNormalization(4096))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(4096, opt and opt.nClasses or 1000))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():add(features):add(classifier)
   return model
end

return createModel
