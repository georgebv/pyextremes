## Disclaimer
**pyextremes** was created to make the process of running EVA simpler
and faster. While the project is built with reasonable defaults which produce
good results in most cases, one should not rely on the model as a source
of ground truth. It is always the responsibility of the user to understand
the subject of analysis and to properly interpret the model outputs.

???+ example
    A 1000-year wave height of 50 meters is not physical and
    is an artifact of the underlying statistical model.
    One should always be mindful of the actual process being modeled and
    remember that the model gives a proabilistic estimate of extreme values
    under the assumption that the selected model (e.g. GEVD or GPD)
    correctly describes the real process.

## Tutorial Structure
**pyextremes** consists of individual components (e.g. extreme value
extraction, model fitting, visualization, etc.) which can be used to perform
a variety of tasks related to EVA. In order to make life easier, these components
were grouped together in a helper class `EVA` (located in `pyextremes.eva.EVA`)
to implement the most common workflow of EVA. Because of this, almost every problem
can generally be solved using either `EVA` or one or more individual components.
This allows for the library to be flexible and makes it easy to integrate into
custom workflows.

Each tutorial section covers a particular area of EVA, such as extreme value
extraction. Solutions are presented both using the `EVA` class and individual component
used by the `EVA` class behind the scenes. Users should be aware that the `EVA` class
shown earlier is not the only way to use pyextremes and that learning the library
would allow for more flexibility when doing analysis.

???+ example
    You can use `pyextremes.get_extremes`,
    `pyextremes.extremes.block_maxima.get_extremes_block_maxima`, or
    `pyextremes.extremes.peaks_over_threshold.get_extremes_peaks_over_threshold`
    functions instead of the `EVA.get_extremes` method to extract extreme values.
    Those are fully equivalent and, in fact, the `EVA` class uses the
    `pyextremes.get_extremes` function (which, in turn, usese the other two)
    behind the scenes.

--8<-- "docs/includes/abbreviations.md"
