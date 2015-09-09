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

An obvious first step is to compute a few summary statistics. The next step would be to create a histogram, in which we group our data into discrete *buckets* and count how many points fall into each bucket. 

```python
def bucketize(point, bucket_size):
	"""floor the point to the next lower multiple of bucket_size"""
	return bucket_size * math.floor(point / bucket_size)

def make_histogram(pints, bucket_size):
	return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=''):
	histogram = make_histogram(points, bucket_size)
	plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
	plt.title(title)
	plt.show()
```

#### Many dimensions
With many dimensions, we'd like to know how all the dimensions relate to one another. A simple approach is to look at the *correlation* matrix, in which the entry in row `i` and column `j` is the correlation between the `i`th dimension and the `j`th dimension of the data:

### Cleaning and munging

We can create a function that wraps `csv.reader`. We'll give it a list of parsers, each specifying how to parse one of the columns. We use `None` to represent "don't do anything to this column":

https://github.com/joelgrus/data-science-from-scratch/blob/master/code/working_with_data.py#L110-L131

### Manipulating data

> [Data Science from Scratch], p130

We will create a function to pick a field out of a `dict`, and another function to pluck the same field out of a collection of `dict`s:
```python
def picker(field_name):
	return lambda row: row[field_name]

def pluck(field_name, rows):
	return map(picker(field_name), rows)
```
This trick for using `map` is clever. 

