#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include <complex.h>
#include "fp_eif.c"
#include "fpt_eif.c"
#include "lr_eif.c"
#include "mc_eif_fixed.c"
#include "mc_eif_rand.c"

static PyMethodDef RNNMethods[] = {
    {"mc_eif_fixed", mc_eif_fixed, METH_VARARGS, "Python interface for EIF network in C"},
    {"mc_eif_rand", mc_eif_rand, METH_VARARGS, "Python interface for EIF network in C"},
    {"fp_eif", fp_eif, METH_VARARGS, "Python interface for EIF network in C"},
    {"lr_eif", lr_eif, METH_VARARGS, "Python interface for EIF network in C"},
    {"fpt_eif", fpt_eif, METH_VARARGS, "Python interface for EIF network in C"},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef _rsnn = {
    PyModuleDef_HEAD_INIT,
    "_rsnn",
    "Python interface for IF network in C",
    -1,
    RNNMethods
};

PyMODINIT_FUNC PyInit__rsnn(void) {
    import_array();
    return PyModule_Create(&_rsnn);
}
