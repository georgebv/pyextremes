## Disclaimer
**pyextremes** was created to make the process of running EVA simpler
and faster. While the project is built with reasonable defaults which produce
good results in most cases, one should not rely on the model as a source
of ground truth. It is always the responsibility of the user to understand
what the process does and to properly interpret the model outputs.
Classic principle applies: garbage in - garbage out.

For example, a 1000-year wave height of 50 meters is not physical and
is an artifact of the underlying statistical model.
One should always be mindful of the actual process being modeled and
remember that the model gives a proabilistic estimate of extreme values
under the assumption that the selected model (e.g. GEVD or GPD)
fully describes the real process.

## How to Read This Tutorial
**pyextremes** was built with an effort to create a flexible and extensible
framework which can be used in part or fully, either standalone or
integrated into another pipeline. Each section within this tutorial
describes a particular step within EVA, such as extreme value extraction,
model fitting, or model performance evaluation. Typically, each of these
processes are implemented via independent components which can be used
on their own to solve a particular task (e.g. extract extreme values).
However, usually EVA is performed as a sequence of the same actions:
extract extremes -> fit model -> get return values.
In order to make this process easier, a helper class `EVA`
(located in `pyextremes.eva.EVA`) was created - this class implements most
features of this library in a compact and consice way and allows users to perform
most analyses in a few lines of code.

Each tutorial section describes how to solve the same problem by
either using individual components or the `EVA` class where possible.
The intention is to demonstrate that it is possible to take a particular
element of the library and use it in your own model - e.g. you want
to extract the extreme values yourself or you would like to use some
composite model and want **pyextremes** only as a visualization tool.
Individual components also provide better performance than `EVA` itself
in scenarios where you need only specific portion of it - e.g. you
want to perform EVA on thousands of grid points.

--8<-- "docs/includes/abbreviations.md"
