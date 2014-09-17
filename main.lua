require "env"
require "cunn"
require "nngraph"
require "nn"
require "optim"

include "ClassNLLCriterion.lua"

function create_network(params)
  local x = nn.Identity()()
  local y = nn.Identity()()
  local i2h = nn.Linear(params.input, params.size)
  local f = i2h(x)
  local r = nn.ReLU()(f)
  local output = nn.Linear(params.size, params.output)(r)
  local probs = nn.LogSoftMax()(output)
  local err = ClassNLLCriterion()({probs, y}) 
  local mlp = nn.gModule({x, y}, {err})
  mlp:getParameters():uniform(-0.1, 0.1)
  return mlp:cuda()
end

local params = {batch_size=128, size=200, input=5, output=2}
local mlp = create_network(params)
paramx, paramdx = mlp:getParameters()
local x = torch.zeros(params.batch_size, params.input)
local xones = torch.zeros(params.batch_size)
local ones = torch.ones(params.input)
local y = torch.zeros(params.batch_size)

function get_data()
  x:random(2):add(-1)
  xones:zero():add(1)
  xones:mv(x, ones) 
  y:zero()
  for i = 1, y:size(1) do
    y[i] = xones[i] % 2 + 1 
  end 
  return x:cuda(), y:cuda()
end

function eval(paramx_)
  if paramx_ ~= paramx then paramx:copy(paramx_) end 
  paramdx:zero()
  local x, y = get_data()
  local err = mlp:forward({x, y})
  accs[step % 10 + 1] = err[2]
  mlp:backward({x, y}, {1})
  return paramx, paramdx
end

accs = torch.zeros(10)
step = 0
while true do
  optim.sgd(eval, paramx, {learningRate=0.01}, {})
  -- Meta - network for optimization goes here
  step = step + 1
  if step % 10 == 0 then
    print(string.format("Step = %d, accuracy = %f", step, accs:mean()))
  end
end
