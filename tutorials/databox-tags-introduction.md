


> Written with [StackEdit](https://stackedit.io/).
# SpECTRE DataBox
This is a high level introduction to the SpECTRE data structure, `DataBox`, for those who do not have pre-existing knowledge on the topic. The goal of this tutorial is to provide higher-level context before jumping into the more fleshed-out development guides on `DataBox`es. In going forward with development and actually working with `DataBox`es, the official SpECTRE documentation for `DataBox`es should be consulted for comprehensive information, instructions, and requirements. The motivation for `DataBox`es, more detail on Simple and Compute tags, and how to add tags to and remove tags from a `DataBox` can be found [here](https://spectre-code.org/databox_foundations.html), and the in-depth documentation on `DataBox`es can be found [here](https://spectre-code.org/group__databoxgroup). This tutorial relies on and borrows heavily from the first link.

# Introduction to DataBoxes and Tags
### What is a DataBox?
**The short, vague answer:** It is a data structure. It is a flexible container that can hold data you give it and, using it, can compute new data that you ask for.

### What is a DataBox used for?
In SpECTRE executables, you will most often see them associated with a **parallel component**. In short, a parallel component is an entity that can exist in parallel. In SpECTRE, they are the things through which we can calculate and run things in parallel. For more information on parallel components, see [this tutorial](https://github.com/macedo22/spectre/blob/tutorial-temp/tutorials/executables-introduction.md) and the [official SpECTRE development guide on parallelization](https://spectre-code.org/dev_guide_parallelization_foundations.html). More generally, however, they are SpECTRE's way of storing and computing new data. This might include physical quantities given as input or calculated and evolved during simulations, such as black hole masses, spins, horizon curvature, etc.

### What goes in a DataBox?
In order to store data in a `DataBox` or compute new things from that data, we'll need **tags**. You may already be familiar with data structures that represent "pairings" such as dictionaries or maps, where a key is mapped to a value. Tags are similar in that they can be thought of as a pairing, but this pairing is mostly between a data type and value. There are two main types of tags, Simple and Compute, and they will be discussed below.

**Simple tags**

An example comparison between a more familiar data structure involving pairing vs tags follows. Let's say we wanted to store the mass and volume of something, and perhaps other physical quantities later. To do this, we could create a map, where the keys of the map are strings representing those physical attributes and the values are the actual physical quantities. For example, say we had some velocity, radius, density, and volume and wanted to keep it all in one place. We might store this in a map called `naive_databox` like so:
(The below example is borrowed from the official [SpECTRE DataBox motivation documentation](https://spectre-code.org/databox_foundations.html)).
```
// Set up variables:
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;

// Set up databox:
std::map<std::string, double> naive_databox;
naive_databox["Velocity"] = velocity;
naive_databox["Radius"] = radius;
naive_databox["Density"] = density;
naive_databox["Volume"] = volume;
```

Here you can see a `string` like `"Velocity"` is mapped to a `double` that represents some velocity value. In order to do something similar with `Databox`es, we'll need these pairings to be organized into tags. There are two types of tags: Simple and Compute tags. For this part, we'll start with a Simple tag for velocity, and that only requires two ingredients: a `name` function that returns a `string` representing the name of the quantity and a `type` alias that represents the data type of data being stored (e.g. velocity would be a `double`). A Simple tag for volume would look like the following:
```
struct Velocity : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Velocity"; }
};
```

A Simple tag for density would look like the following:
```
struct Density : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Density"; }
};
```

Notice that the `string` returned by the `name` function should match the name of the tag struct and that this Simple tag should inherit from the `DataBox` type `db::SimpleTag`.

**Compute tags**

But what if we want to use values we're storing in our `DataBox` to calculate new quantities? For example, what if we wanted to compute the mass from the volume and density that we already have? This is where Compute tags come in. Compute tags will often inherit from a corresponding Simple tag, but they have an extra piece: an associated function that executes a calculation and returns a value. This function is what enables Compute tags to compute new quantities.

Compute tags require two ingredients: a function named `function` and an alias named `argument_tags` (if they don't inherit from a Simple tag, they also need a `name` function, like the one in the previous example Simple tags above). Let's say that we wanted to create a Compute tag for calculating mass from density and volume. We can imagine that this would require 2 arguments for that function, one for volume and one for density. These can be the input arguments to the `function` that will be part of the tag. We also need `argument_tags` to be a `tmpl::list` of *tags* that correspond to the *arguments* of `function`. In other words, if we have the mass Compute tag's `function` take volume and density as arguments to compute the mass, then the `argument_tags` list must also include the corresponding volume tag and density tag. Below is an example of what a Simple and Compute tag for the mass might look like:

```
// mass simple tag
struct Mass : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Mass"; }

// mass compute tag
struct MassCompute : Mass, db::ComputeTag {
  static double function(
      const double& volume,
      const double& density) noexcept {
    return volume * density;
  }
  using argument_tags = tmpl::list<Volume, Density>;
}
```
Notice that the order of the arguments in `function` are in the same order as the corresponding tags in `argument_tags`. Also notice that the Compute tag for the mass is named by adding on "Compute" to the end of the Simple tag's name and that the Compute tag inherits both from its own corresponding Simple tag (`Mass`) as well as the `DataBox` type `db::ComputeTag`. Lastly, notice that the return type of `function` matches the `type` alias in the Simple tag (they are both `double)`.

Lastly, note that instead of defining and implementing `function` here within the compute tag, a `static constexpr` function pointer that points to an existing function can be used, instead. For example, let's say that some function called `get_mass` was defined inside a namespace called `SomeNamespace` in another file and it looked like so:

```
namespace SomeNamespace {
  double get_mass(
      const double& volume,
      const double& density) noexcept {
    return volume * density;
  }
}
```

Then, our `MassCompute` Compute tag could use a function pointer to this function, provided that the function's `hpp` file was included at the top of the file that the Compute tag was being defined in. It would instead look like the following:
```
// mass compute tag using function pointer
struct MassCompute : Mass, db::ComputeTag {
  static constexpr auto function = &SomeNamespace::get_mass;
  using argument_tags = tmpl::list<Volume, Density>;
}
```

**Summary of tags:**
- there are two main types: Simple and Compute
- they are added to `DataBoxes` to store and compute new quantities
- Compute tags will often inherit from corresponding Simple tags
- Compute tags are used for when you want to compute new values from data you already have in your `DataBox`. This means that if you add a Compute tag to your `DataBox`, you will need to make sure that all tags that your Compute tag depends on are already in the `DataBox` as Simple or Compute tags, as well - otherwise, there will be no way to use the `function` associated with your Compute tag (e.g. if you added the `MassCompute` Compute tag to a `DataBox`, that `DataBox` better also already have the `Volume` and `Density` Simple or Compute tags). This also means that you'll need to check further back if those arguments are dependent on other tags and include those, as well.

### Why are DataBoxes and tags used?
If we can just use a map or a dictionary to map physical quantities to their values, why all the run-around to create a `Databox`? The major advantage of using `DataBox`es is that the compiler can do type checking on tags at compile time. Instead of something like velocity being a `double` associated with a `string` in a map at run time, it is a type that is evaluated and checked at compile time. The advantage of this is that we're moving work that would normally be done during run time to the compiler. This makes compiling take longer, but in this way, more checks can be done during compile time to avoid issues during run time, and moving more work to the compiler also means less work during run time. For a more fleshed out discussion on the motivation for `DataBox`es in lieu of other data structures like maps, see [this](https://spectre-code.org/databox_foundations.html).

### How do I create a `DataBox`, add or remove tags from one, or work with them?
See the two SpECTRE documentation links at the top of this tutorial.
