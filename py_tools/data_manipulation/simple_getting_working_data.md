<!-- toc -->

# Simple Getting and Working with Data
 > start: 31 August 2015
 > Adapted from [Data Science from Scratch], Chapter 9

## Getting Data

### `stdin` and `stdout`
When running at the command line, we can *pipe* data through using `sys.stdin` and `sys.stout`. 

https://github.com/joelgrus/data-science-from-scratch/blob/master/code/egrep.py

We can also count the lines it receives:
```python
import sys
count = 0
for line in sys.stdin:
	count += 1
print(count)
```
We can then use it in terminal:
```bash
cat SomeFile.txt | python egrep.py "0-9" | python line_count.py
```

### Reading Files

```python
file_for_reading = open('reading_file.txt', 'r')

file_for_writing = open('writing_file.txt', 'w')

file_for_appending = open('appending_file.txt', a)

file_for_writing.close()
```

Since it is easy to forget to close the files (as we are lazy programmers! yeah), we should use it in a `with` block:

```python
with open(file_name, 'r') as f:
	data = function_that_gets_data_from(f)
# at this point f has already been closed, so don't try to use it
process(data)
```

---

If we need to read a whole text file, we can just iterate over the lines of the file using `for`:
```python
starts_with_hash = 0

with open('input.txt', 'r') as f:
	for line in f:
		if re.match("^#", line):
			starts_with_hash += 1
```

## Working with Data

### Exploring Data
After identifying the questions we're trying to answer and have gotten some data, we next step should be to *explore* our data. 

#### Exploring one-dimensional data

