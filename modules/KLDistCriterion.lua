--[[

    Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images
    by Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, Martin Riedmiller
    http://arxiv.org/abs/1506.07365

    John-Alexander M. Assael (www.johnassael.com)
    Implementation copyright (C) 2015

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

local KLDistCriterion, parent = torch.class('nn.KLDistCriterion', 'nn.Criterion')

function KLDistCriterion:__init()
   parent.__init(self)
   self.sizeAverage = false
end

function KLDistCriterion:updateOutput(input, target)
    -- minimize, 0 when p = q
    -- input = q, target = p
    -- input[2] = log(sigma^2)

    local q = {mu=input[1][1]:clone(), sigma=input[2][1]:clone():exp()} -- :sqrt()
    local p = {mu=target[1][1]:clone(), sigma=target[2][1]:clone():exp()} --:sqrt()
    local ndim = q.mu:nElement()

    -- Determinants of diagonal covariances pv, qv
    -- dpv = p.sigma:clone():prod()
    -- dqv = q.sigma:clone():prod()

    -- Inverses of diagonal covariances pv, qv
    iqv = torch.ones(ndim):cdiv(q.sigma)
    -- ipv = torch.ones(ndim):cdiv(p.sigma)

    -- Difference between means pm, qm
    diff = q.mu - p.mu


    self.output = -(  --torch.log(dqv) - torch.log(dpv)                    -- log |\Sigma_q| / |\Sigma_p|
                     torch.log(q.sigma):sum() - torch.log(p.sigma):sum()                    -- log |\Sigma_q| / |\Sigma_p|
                     + torch.cmul(iqv, p.sigma):sum()                 -- + tr(\Sigma_q^{-1} * \Sigma_p)
                     -- + torch.cmul(diff, torch.cmul(iqv, diff)):sum()  -- + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
                     + torch.cmul(diff:clone():pow(2), iqv):sum()  -- + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
                     - ndim) / 2                        -- - N

    return self.output
end

function KLDistCriterion:updateGradInput(input, target)    
    
    -- logdetX^{−1}=log(detX)^{−1}=−logdetX

    -- local q = {mu=input[1][1]:clone(), sigma=input[2][1]:clone():exp()}
    -- local p = {mu=target[1][1]:clone(), sigma=target[2][1]:clone():exp()}
    local q = {mu=input[1][1]:clone(), sigma=input[2][1]:clone()}
    local p = {mu=target[1][1]:clone(), sigma=target[2][1]:clone()}
    local ndim = q.mu:nElement()

    self.gradInput = {}
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(input[1])

    self.gradInput[2] = self.gradInput[2] or input[1].new()
    self.gradInput[2]:resizeAs(input[1])

    -- Inverses of diagonal covariances pv, qv
    iqv = torch.exp(-q.sigma)
    -- ipv = torch.ones(ndim):cdiv(p.sigma)

    -- Difference between means pm, qm
    diff = q.mu - p.mu

    -- d q_mu iqv (-pmu + qmu)
    self.gradInput[1] = torch.cmul(-diff, iqv):view(1, ndim)

    -- d q_sigma -((psigma2 + (pmu - qmu)^2 - qsigma2)/(2 qsigma2^2))

    -- self.gradInput[2]:copy(p.sigma):add(diff:clone():pow(2)):add(-1, q.sigma):cmul(iqv:clone():pow(2)):mul(-0.5):view(1, ndim)
    self.gradInput[2]:copy(torch.exp(p.sigma)):add(-1, torch.exp(q.sigma)):add(torch.pow(diff, 2)):cmul(iqv):mul(0.5):view(1, ndim)

    return self.gradInput
end