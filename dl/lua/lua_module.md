<!-- toc -->

@(Cabinet)[ml_dl_lua|published_gitbook]

# Understanding Neural Net Module with Lua/Torch

> date: 2016-02-17

> http://code.madbits.com/wiki/doku.php?id=tutorial_morestuff

`module` is an abstract class which defines fundamental methods necessary for training a neural network. All `module`s are serializable. 

`module`s contain two states variables: `output` and `gradInput`. 

### [output] forward(input)
Takes an input object, and computes the corresponding output of the module. 

After a `forward()`, the output state variable should have been updated to the new state. 

We do NOT override this function. Instead, we implement `updateOutput(input)` function. The `forward` module in the abstract parent class `module` will call `updateOutput(input)`. 

### [gradInput] backward(input, gradOutput)
Performs a backpropagation step through the module, w.r.t. the given input. 

A backpropagation step consists of computing **two** kind of gradients at input given `gradOutput` (gradients w.r.t. the output of the module). This function simply performs this task using two function calls:
1. a function call to `updateGradInput(input, gradOutput)`
2. a function call to `accGradParameters(input, gradOutput)`

We do NOT override this function call. We override `updateGradInput(input, gradOutput)` and `accGradParameters(input, gradOutput)` functions.


### [output] updateOutput(input, gradOutput)
When defining a new module, this method should be overloaded. 

Computes the output using the current parameter set of the class and input. This function returns the result which is stored in the output field. 

### [gradInput] updateGradInput(input, gradOutput)
When defining a new module, this method should be overloaded. 

Computes the gradient of the module w.r.t. its own input. This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

### [gradInput] accGradParameters(input, gradOutput)
When defining a new module, this method should be overloaded, if the module has **trainable parameters**. 

Computes the gradient of the module w.r.t. its own parameters. Many modules do NOT perform this step as they do NOT have any **trainable parameters**. The module is expected to accumulate the gradients w.r.t. the **trainable parameters` in some variables. 

Zeroing this accumulation is achieved with `zeroGradParameters()` and updating the **trainable parameters** according to this accumulation is done with `updateParameters()`. 

---

## New Module Class Template
The following is an empty holder for a typical new class using `torch.class`:

```lua
require 'torch'
local NewClass, Parent = torch.class('nn.NewClass', 'nn.Module')

function NewClass:__init()
	parent.__init(self)
end

function NewClass:updateOutput(input)
end

function NewClass:updateGradInput(input, gradOutput)
end

function NewClass:accGradParameters(input, gradOutput)
end

function NewClass:reset()
end
```

## Testing Correctness
When writing modules with gradient estimation, it's always very important to test your implementation. This can be easily done using the `Jacobian` class provided in `nn`, which compares the implementation of the gradient methods (`updateGradInput()` and `accGradParameters()`) with the Jacobian matrix obtained by finite differences (perturbating the input of the module, and estimating the deltas on the output). This can be done like this:

```lua
-- parameters
local precision = 1e-5
local jac = nn.Jacobian
 
-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.5
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.Dropout(percentage)
 
-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
```

One slight issue with the `Jacobian` class is the fact that it assumes that the outputs of a module are deterministic w.r.t. the inputs. This is not the case for that particular module, so for the purpose of these tests we need to freeze the noise generation, i.e. do it only once:

```lua
-- we overload the updateOutput() function to generate noise only
-- once for the whole test.
function nn.Dropout.updateOutput(self, input)
   self.noise = self.noise or torch.rand(input:size()) -- uniform noise between 0 and 1
   self.noise:add(1 - self.p):floor()  -- a percentage of noise
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   return self.output
end
```