> Written with [StackEdit](https://stackedit.io/).
# SpECTRE parallel executables

This is a high level introduction to the structure and ingredients of a SpECTRE parallel executable using an existing example executable, for those who do not have pre-existing knowledge on the topic. The goal of this tutorial is to provide higher-level context for some SpECTRE concepts and how they connect in a parallel executable before jumping into the more fleshed-out development guides. In going forward with development, the official SpECTRE [creating executables guide](https://spectre-code.org/dev_guide_creating_executables.html) and [parallelization guide](https://spectre-code.org/dev_guide_parallelization_foundations.html) should be consulted for comprehensive information, instructions, and requirements.


# What ingredients go into a SpECTRE parallel executable and how are they related?

**The main ingredients include:**
- Phases: temporal chunks of the executable
- Actions: computational actions (e.g. calculating a value, printing a message)
- Parallel components: units on which Actions can be done in parallel
- Metavariables: high-level configuration for the compiler regarding the system to simulate
- Input file: a yaml file containing options for the executable

**How they are related:**

Temporally, a parallel executable is broken up into Phases. During each Phase, certain Actions will be executed on certain parallel components. Each parallel component  has a `phase_dependent_action_list` that defines the list of Phase-dependent Actions that are to be carried out in which Phase on that parallel component. The Metavariables struct defines, among other things, the parallel components and Phases that exist in the executable and how the executable should proceed from one Phase to the next. When you run an executable, it takes an input file as a command line argument that specifies options called OptionTags that are used to initialize parallel components with initial input data.

The sections below provide some more notes regarding each of these ingredients, but this is in no way comprehensive.

# Walkthrough of SingletonHelloWorld
We will use the <code>SingletonHelloWorld</code> executable as an example to step through to see how these pieces fit together to make an executable. The source file is [here](https://github.com/sxs-collaboration/spectre/blob/develop/src/Executables/Examples/HelloWorld/SingletonHelloWorld.hpp). The source input file is [here](https://github.com/sxs-collaboration/spectre/blob/develop/tests/InputFiles/ExampleExecutables/SingletonHelloWorld.yaml).

## Metavariables breakdown

To see what this executable does and which Phases it defines, let's check out its Metavariables struct, defined as `Metavars` here:
```
struct Metavars {
  using const_global_cache_tag_list = tmpl::list<>;

  using component_list = tmpl::list<HelloWorld<Metavars>>;

  static constexpr OptionString help{
      "Say hello from a singleton parallel component."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};
```
The `OptionString` tells us the overall purpose of the executable: to print some "hello" message.

The `component_list` is the list of parallel components that exist in this executable. These are units through which we can execute Actions in parallel. In this case, the list only contains one parallel component called `HelloWorld`.

The `Phase` class tells us that this executable has three Phases: `Initialization`, `Execute`, and `Exit`. Each executable must have at least an `Initialization` and `Exit` Phase, but can define additional Phases.

The `determine_next_phase` function determines which Phase should be run next after one finishes. In this case, it says that when `Initialization` is complete, then enter the `Execute` Phase, and when that's complete, enter the `Exit` Phase.

**Metavariables takeaway:** We have an executable that defines one parallel component called `HelloWorld`, three Phases called `Initialization`, `Execute`, and `Exit`, and altogether the executable should print some "hello" message.

## Parallel component breakdown

But what happens during each of these Phases defined in `Metavars`? Each parallel component has a list of Phase-dependent Actions (PDALs) that will be executed during certain phases. It will also have an `execute_next_phase` function that says what to do when each Phase outside of `Initialization` and `Exit` completes. `HelloWorld` is the only parallel component defined for this executable, so let's check out its struct:

```
struct HelloWorld {
  using const_global_cache_tag_list = tmpl::list<OptionTags::Name>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using options = tmpl::list<>;
  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /* global_cache */) noexcept {}
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept;
};
```
The `chare_type` here says that `HelloWorld` is a singleton component, which means that only one such `HelloWorld` object will exist in parallel across processors.

The `const_global_cache_tag_list` defines the list of OptionTags needed by `HelloWorld`. It only lists one such tag, `Name`. This is referring to the one option specified by the [sample input yaml file](https://github.com/sxs-collaboration/spectre/blob/develop/tests/InputFiles/ExampleExecutables/SingletonHelloWorld.yaml), where the option is specified by the line:

```
Name: Albert Einstein
```
OptionTags will not be covered in this tutorial, but the takeaway here is that this option is a piece of information that is constant and will be associated with the component so that we can grab it later and use it.

The `initialize` function is called during the `Initialize` Phase of the executable, but this component's is empty, which means it doesn't do anything during the `Initialization` Phase.

The `phase_dependent_action_list` tells us which Phase-dependent Actions are executed in which Phases for this `HelloWorld` component. This is a _list of lists_, not literally but in a sense. It is a `tmpl::list` of `PhaseActions` elements, where each `PhaseActions` element in the list has a Phase and the Phase-dependent Action list (PDAL) for that Phase, which are specified by the second and third template arguments, respectively. This `tmpl::list` of `PhaseActions` only has one element, and that `PhaseActions` element specifies that during the `Execute` Phase, there are no Phase-dependent Actions to take on the component (i.e. the third template argument is an empty `tmpl::list` of Actions). In short, this parallel component will not have any Phase-dependent Actions performed on it. Then why do we even have the one `PhaseActions` element for the `Execute` Phase if its PDAL is empty (why isn't `phase_dependent_action_list` an empty `tmpl::list`)? This is because if a parallel component has no Phase-dependent Actions whatsoever, it still must specify at least one `PhaseActions` element, so we just leave its PDAL empty.

The `execute_next_phase` tells us what happens at the end of every Phase outside of `Initialization` and `Exit`, which means that in this executable, it is only called at the end of the `Execute` Phase. This function is declared here in the struct, but defined a bit lower. Here is what the function definition looks like:
```
template <class Metavariables>
void HelloWorld<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::PrintMessage>(
      Parallel::get_parallel_component<HelloWorld>(
          *(global_cache.ckLocalBranch())));
}
```
This function body tells us that all this function does is execute the simple Action called `PrintMessage` that is defined in the `Actions` namespace.  An important thing to note about simple Actions is that they can be called during any Phase.

**HelloWorld parallel component takeaway:** This is a singleton component, so only one `HelloWorld` object will exist and it is associated with the OptionTag called `Name`, whose value is `Albert Einstein`. Nothing happens to the component during the `Initialization` Phase and no Phase-dependent Actions are executed on it. When the `Execute` Phase ends, `execute_next_phase` will execute the `PrintMessage` Action.

## PrintMessage Action breakdown
Since the `HelloWorld` component is the only parallel component and it does not execute any Phase-dependent Actions, the only Action that is run in this whole executable is `PrintMessage`. Let's take a look at its struct, which is defined in the `Actions` namespace, a convention followed by other Actions in other executables:
```
struct PrintMessage {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::printf("Hello %s from process %d on node %d!\n",
                     Parallel::get<OptionTags::Name>(cache),
                     Parallel::my_proc(), Parallel::my_node());
  }
};
```
Each Action has an `apply` method that is called when the Action is executed. This one just prints a "hello" message that includes the `Name` that was specified by the input file, and the processor and node that the Action is being executed on. `Parallel::printf` is used instead of the standard `printf` because it is safe to run in parallel.

We've now stepped through the whole executable!

## Beyond this example: DataBox
`DataBox`es are an important SpECTRE data structure and concept that were not addressed by this example, but are essential to most executables. Each parallel component has an associated `DataBox` that is a flexible container that holds the information you ask for, specified by `SimpleTag`s and `ComputeTag`s. The motivation for `DataBox`es is outlined [here](https://spectre-code.org/databox_foundations.html) and a guide to using and manipulating them is [here](https://spectre-code.org/group__databoxgroup).
