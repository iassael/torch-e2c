--[[

    Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images
    by Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, Martin Riedmiller
    http://arxiv.org/abs/1506.07365

    John-Alexander M. Assael (iassael@gmail.com) and Marc P. Deisenroth.
    Impementation Copyright (c) 2015, All rights reserved.

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

]]--

local AddCons, parent = torch.class('nn.AddCons', 'nn.Module')

function AddCons:__init(constant_scalar,ip)
  parent.__init(self)
  --assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
  
  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function AddCons:updateOutput(input)
  if self.inplace then
    input:add(self.constant_scalar)
    self.output = input
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:add(self.constant_scalar)
  end
  return self.output
end 

function AddCons:updateGradInput(input, gradOutput)
  if self.inplace then
    self.gradInput = gradOutput
    -- restore previous input value
    input:add(-self.constant_scalar)
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  return self.gradInput
end