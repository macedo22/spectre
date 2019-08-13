


> Written with [StackEdit](https://stackedit.io/).
# C++ Object-oriented programming concepts to know for SpECTRE

This is meant to be an introduction to some object-oriented concepts found and used in SpECTRE, but is in no way an exhaustive breakdown or list of all OOP techniques.

## Templates
Normally when writing functions, we specify the data types of arguments and the return value. What templates allow us to do is enable flexibility of data types when it comes to templated functions, templated classes, templated structs, etc.

For example, let's say we wanted to make a simple function that prints the sum of two numbers. We could define such a function like so:
```
void printSumOfTwoInts(int a, int b) {
    std::cout << a + b << std::endl;
}
```
Simple enough. But what if we now wanted to same functionality but with doubles? Then we could do the following:
```
void printSumOfTwoDoubles(double a, double b) {
    std::cout << a + b << std::endl;
}
```
Also simple! But what if we wanted to be able to print the sum of an `int` and a `double`? Then we would have to design a 3rd function that looks almost exactly the same! And what if we wanted to pass in a `double` first and the `int` second? That would be a fourth function! You can see that we'd be repeating the same work over and over to achieve the same functionality. Because we know an operation like `a + b` is legal whether `a` and `b` are `int`s or `double`s, so let's define a templated function:
```
template <typename A, typename B>
void printSumOfTwoNumbers(A a, B b) {
    std::cout << a + b << std::endl;
}
```
By adding `template <typename A, typename B>` above the function, we say that `printSumOfTwoNumbers` is templated on `A` and `B`. Notice that instead of `int` or `double` being the type of arguments `a` or `b`, we say that the type of `a` is `A` and the type of `b` is `B`. When we call this function with appropriate data types for the template parameters like `int` or `double`, those data types will be "filled in" as the types of `a` and `b` for the function. For example, consider the following calls to the function:
```
int one = 1;
int two = 2;
double three = 3.0;
double four = 4.0;

printSumOfTwoNumbers<int, int>(one, two);
printSumOfTwoNumbers<double, double>(three, four);
printSumOfTwoNumbers<int, double>(one, four);
printSumOfTwoNumbers<double, int>(three, two);
```
In the above code snippet, we call the `printSumOfTwoNumbers` function three times, each with a different combination of template parameters. Where you see `<int, double>`, you can think of it as "passing in" the `int` type to be `A`, the type for the argument `a`, and the `double` type to be `B`, the type for the argument `b`.  Instead of having to write *almost* the same function four different times to take a different combination of arguments for `int` and `double`, we were able to define just one templated function that can do the job.

## Inheritance
Inheritance describes a unidirectional "is a" relationship between two classes or structs. For example, we can say that a dog *is an* animal. A cat *is an* animal. A bird *is an* animal. However, it wouldn't be right to say that an animal *is a* dog/cat/bird. In object-oriented programming, this concept is formalized into a parent class (or superclass) and a child class (or subclass), and a parent class can have multiple child classes. For example, let's say we had an `Animal` class that just had a constructor and one private variable, `name`, which is the name of the animal:

```
class Animal {
    private:
        std::string name;
    public:
        Animal(std::string n) {
            name = n;
        }
}
```
```
class Dog :  public Animal {

}
```

## Function overloading

## Aliases
