require "env"
require "cunn"
require "nngraph"
require "nn"
require "optim"

include "ClassNLLCriterion.lua"

function create_network(params)
  local x = nn.Identity()()
  --local y = nn.Identity()()
  --local i2h = nn.ReLU()(nn.Linear(params.input, params.size))
  local i2h = nn.Linear(params.input, params.size)
  local f = i2h(x)
  local mlp = nn.gModule({x}, {f})
--[[
  local output = nn.Linear(params.size, params.output)(f)
  local probs = nn.LogSoftMax()(output)
  local err = ClassNLLCriterion()({probs, y})
  local mlp = nn.gModule({x, y}, {err})
  --mlp:getParameters():uniform(-0.1, 0.1)--]]
  return mlp:cuda()
end

local params = {batch_size=128, samples=1000, size=200, input=10, output=2}
local mlp = create_network(params)
--[[
paramx, paramdx = mlp:getParameters()
local x = torch.zeros(params.batch_size, params.input)
local xones = torch.zeros(params.batch_size)
local ones = torch.ones(params.input)
local y = torch.zeros(params.batch_size, params.output)

function get_data()
  x:random(2):add(-1)
  xones:mv(x, ones) 
  y:zero()
  for i = 1, y:size(1) do
    y[i, xones[i] % 2 + 1] = 1
  end
  return x, y
end

function eval(paramx_)
  local x, y = get_data()
  local err = mlp:forward({x, y}) 
  mlp:backward({x, y})({1})
  return err, paramdx
end

while true do
  err = optim.sgd(eval, x, {learningRate=0.1})
  print(err[0])
end

]]--
