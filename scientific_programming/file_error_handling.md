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

