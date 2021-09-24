## Disclaimer
**pyextremes** was created to make the process of running EVA simpler
and faster. While the project is built with reasonable defaults which produce
good results in most cases, one should not rely on the model as a source
of ground truth. It is always the responsibility of the user to understand
the subject of analysis and to properly interpret the model outputs.

!!! example
    A 1000-year wave height of 100 meters is not physical and
    is an artifact of the underlying statistical model.
    One should always be mindful of the actual process being modeled and
    remember that the model gives a proabilistic estimate of extreme values
    under the assumption that the selected model (e.g. GEVD or GPD)
    correctly describes the underlying process (in this example, water waves).

## Tutorial Structure
Each tutorial section covers a particular area of EVA, such as extreme value
extraction, fitting a model, or summarizing and visualizing analysis results.
**pyextremes** was built in a modular fashion where each of these components
is implemented independently and can be used on its own. In order to make life easier
a helper class `EVA` was created (located in `pyextremes.eva.EVA`) which chains these
components together to streamline the most common types of EVA workflows and to reduce
the amount of code a user needs to write when performing analysis.

When posstible, sections of this tutorial present two alternative ways to perform the
same action: via `EVA` and via low-level functions which are using by `EVA` behind the
scenes.

--8<-- "docs/includes/abbreviations.md"
