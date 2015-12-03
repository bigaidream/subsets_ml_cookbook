<!-- toc -->

@(Cabinet)[ml_general_code|soft_engineering]

date = "2015-12-03"

# Test-Driven Machine Learning, Ch1

> largely taken from the book, [Test-Driven Machine Learning]

Kent Beck says there are two rules in TDD:
* Don't write a line of new code unless you first have a failing automated test
* Eliminate duplication

as the mantra of TDD: "Red, Green, Refactor."

More concretely, TDD has the following 3 processes:

* The writing of code to detect the intended behavioral change.
* A rapid iteration cycle that produces working software after each iteration.
* Clear definitions of what a bug is. If a test is not failing but a bug is found, it is not a bug. It is a new feature. 

TDD can and does operate at many different levels of the software under development. Tests can be written against functions and methods, entire classes, programs, web services, neural networks and whole machine learning pipelines. 

## The TDD cycle

**Red**
Create a failing test. We need to know what failure looks like. 

At the highest level in machine learning, this might be a baseline test where baseline is a "better than random" test. 

**Green**
After you have established a failing test, you can start working to get it green. 

Just slap something together. You will be able to fix the issues in the next step. 

**Refactor**
Refactor specifically means to change your software without affecting its behavior. 

## Behavior-driven development
BDD is the addition of business concerns to the technical concerns more typical of TDD. 

Dan North noticed some issues with TDD:
* People had a hard time understanding what they should test next. 
* Deciding what to name a test could be difficult.
* How much to test in a single test always seemed arbitrary. 

Simply put, BDD is about writing our tests in such a way that they will tell us the kind of behavior change they affect. A good litmus test might be asking oneself if the test you are writing would be worth explaining to a business stakeholder (or rubber duck...). It follows a structure of `Given, When, Then`. A concrete example could be: "`Given` an empty dataset `when` the classifier is trained, it should `then` throw an invalid operation exception".

The BDD adherents tend to use specialized tools to make the language and test result reports be as accessible to business stakeholders as possible. It turns out that this extra elegance is typically used so little that it doesn't seem worthwhile. 