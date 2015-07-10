<!-- toc -->

# File and Error Handling

## File

### Reading files

```python
with open('data.txt', 'r') as infile:
	for line in infile:
		# process line
```
It is equivalent to 
```python
infile = open('data.txt', 'r')
for line in infile:
	# process line
infile.close()
```

## Handling errors

### Raising exceptions
```python
import sys

def read_C():
    try:
        C = float(sys.argv[1])
    except IndexError:
        raise IndexError\
        ('Celsius degrees must be supplied on the command line')
    except ValueError:
        raise ValueError\
        ('Celsius degrees must be a pure number, '\
         'not "%s"' % sys.argv[1])
    # C is read correctly as a number, but can have wrong value:
    if C < -273.15:
        raise ValueError('C=%g is a non-physical value!' % C)
    return C

try:
    C = read_C()
except (IndexError, ValueError), e:
    print e
    sys.exit(1)
    
F = 9.0*C/5 + 32
print '%gC is %.1fF' % (C, F)
```

## How to make Python find a module file

The most easy way is to put the `module` in the same folder as the caller. 

Otherwise, Python looks for modules in the folders contained in the list `sys.path` 
```python
import sys, pprint
pprint.pprint(sys.path)
```

We can now do one of the two things:
1. Place the module file in one of the folders in `sys.path`.
2. Include the folder containing the module file in `sys.path`.

There are two ways of doing the latter task. Alternative 1 is to explicitly insert a new folder name in `sys.path` in the program that uses the module:
```python
modulefolder= '../../pymodules'
sys.path.insert(0, modulefolder)
```

Alternative 2 is to specify the folder name in the `PYTHONPATH` environment variable. All folder names listed in `PYTHONPATH` are automatically included in `sys.path` when a Python program starts. 

-----
In `PyCharm`, we can do it by going to `File ->  Settings -> Project Structure` selecting the `mymodule` folder in the tree and clicking `sources` to add it. 

http://stackoverflow.com/questions/10909857/how-can-i-get-pycharm-to-find-the-right-paths-if-i-open-a-directory-that-is-not
