.. include:: sinebow.rst

.. raw:: html

        <h1>
            <span style="color: #e72333">Pykonal</span>
        </h1>



Pykonal is a library which solves the Eikonal Function for 2D and 3D instance masks, using the Fast Iterative Method.
It achieves memory efficiency through a fused kernel written in OpenAI Triton.
It also provides functionality to calculate gradients of the eikonal field.

This implementation uses convolutions to calculate affinity masks, which may be faster and
can occur on cuda, however uses dense representations of the affinity masks and therefore is
memory intensive. It may be possible for much of the mask operations to occur with sparse tensors
for memory efficiency.

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2

            .. grid-item-card::

                .. toctree::
                    :caption: Basics
                    :maxdepth: 1

                    quickstart


    .. grid-item::

        .. grid:: 1 1 1 1
            :gutter: 2

            .. grid-item-card::

                .. toctree::
                    :caption: API Reference
                    :maxdepth: 2

                    api_reference.rst