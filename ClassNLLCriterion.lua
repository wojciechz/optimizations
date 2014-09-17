local ClassNLLCriterion, parent = torch.class('ClassNLLCriterion', 'nn.Module')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function ClassNLLCriterion:updateOutput(input)
  local acc = 0 
  local input, target = unpack(input)
  local output = 0 
  for i=1,target:size(1) do
    output = output - input[i][target[i]]
    if input[i]:max() == input[i][target[i]] then
      acc = acc + 1 
    end 
  end 
  if self.sizeAverage then
    output = output / target:size(1)
    acc = acc / target:size(1)
  end 
  self.output = {output, acc}
  return self.output
end

function ClassNLLCriterion:updateGradInput(input)
  local input, target = unpack(input)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()

  if input:dim() == 1 then
    self.gradInput[target] = -1
  else
    local z = -1
    if self.sizeAverage then
      z = z / target:size(1)
    end 
    local gradInput = self.gradInput
    for i=1,target:size(1) do
      gradInput[i][target[i]] = z 
    end 
  end 
  return self.gradInput
end
