***************
Getting Started
***************

Installation
============
The VASP step is installed with the `SEAMM Manager`_, and is probably already part of
your SEAMM installation. To add it, or bring it up to date::

  seamm-manager install vasp-step
  seamm-manager update vasp-step

or use the Manager's window. VASP itself is licensed software that you install
yourself, so installing the step does not install VASP. Instead it creates
``~/SEAMM/vasp.ini`` -- or ``vasp.ini`` in whichever SEAMM installation you are
working on -- which tells SEAMM where VASP is and how to run it. An existing file
is never changed.

.. _SEAMM Manager: https://molssi-seamm.github.io/getting_started/installation/seamm-manager.html

Configuring VASP
================
Edit the ``[local]`` section of ``~/SEAMM/vasp.ini`` to give the command lines for the
three VASP executables. ``{NTASKS}`` is replaced by the number of MPI tasks SEAMM
chooses for the calculation:

.. code-block:: ini

    [local]
    installation = local
    code = mpiexec -np {NTASKS} /path/to/vasp/bin/vasp_std
    gamma_code = mpiexec -np {NTASKS} /path/to/vasp/bin/vasp_gam
    noncollinear_code = mpiexec -np {NTASKS} /path/to/vasp/bin/vasp_ncl

``code`` is the standard build, used for most calculations; ``gamma_code`` is used when
only the Gamma point is sampled, and ``noncollinear_code`` for non-collinear spin.
Without a Gamma-only build SEAMM uses ``code`` instead, which gives the same results
more slowly; non-collinear calculations need ``noncollinear_code``.

If VASP comes from environment modules, give the modules to load and the executables'
names:

.. code-block:: ini

    [local]
    installation = modules
    modules = VASP/6.5.1-foss-2024a
    code = mpiexec -np {NTASKS} vasp_std
    gamma_code = mpiexec -np {NTASKS} vasp_gam
    noncollinear_code = mpiexec -np {NTASKS} vasp_ncl

Use ``srun -n {NTASKS}`` or whatever your site uses in place of ``mpiexec -np
{NTASKS}``. If the commands are not given but ``vasp_std`` (and ``vasp_gam`` and
``vasp_ncl``) are on your ``PATH``, SEAMM fills them in the first time it runs VASP.
The file itself has comments explaining each option.

That should be enough to get started. For more detail about the functionality in this plug-in, see the :ref:`User Guide <user-guide>`.
