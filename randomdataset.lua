local tntenv = require 'torchnet.env'
local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local RandomDataset, ResampleDataset =
   torch.class('tnt.RandomDataset', 'tnt.ResampleDataset', tntenv)

RandomDataset.__init = argcheck{
   doc = [[
<a name="RandomDataset">
#### tnt.RandomDataset(@ARGP)
@ARGT

`tnt.RandomDataset` is built using ResampleDataset.
Use `:random()` to substitute `:shuffle()` and `:random(size)` to
substitute `:shuffle(size,true)`.

Purpose: uses random sampling so `resample()` does not have to be called
after every epoch. Note: keep seed different if using in Parallel.
]],
   {name='self', type='tnt.RandomDataset'},
   {name='dataset', type='tnt.Dataset'},
   {name='size', type='number', opt=true},
   call =
      function(self, dataset,size)
         local function sampler(dataset, idx)
            return torch.random(1,dataset:size())
         end
         ResampleDataset.__init(self, {
            dataset = dataset,
            sampler = sampler,
            size    = size })
      end
}

tnt.Dataset.random =
   function(...)
      return tnt.RandomDataset(...)
   end
