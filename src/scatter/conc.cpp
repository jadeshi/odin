
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

#ifdef NO_OMP
   #define omp_get_thread_num() 0
#else
   #include <omp.h>
#endif

#include "conc.h"

using namespace std;

/*! TJL 2012 */

#define MAX_NUM_TYPES 10


// "kernel" is the function that computes the scattering intensities
void kernel( float const * const __restrict__ q_x, 
             float const * const __restrict__ q_y, 
             float const * const __restrict__ q_z, 
             float *outQ, // <-- not const 
             int   const nQ,
             float const * const __restrict__ r_x, 
             float const * const __restrict__ r_y, 
             float const * const __restrict__ r_z,
             int   const * const __restrict__ r_id, 
             int   const numAtoms, 
             int   const numAtomTypes,
             float const * const __restrict__ cromermann ) {
            

    // for each q vector
    // #pragma omp parallel for shared(outQ, q0, q1, q2, q3)
    for( int iq = 0; iq < nQ; iq++ ) 
    {
        float qx = q_x[iq];
        float qy = q_y[iq];
        float qz = q_z[iq];

        // workspace for cm calcs -- static size, but hopefully big enough
        float formfactors[MAX_NUM_TYPES];

        // Cromer-Mann computation, precompute for this value of q
        float mq = qx*qx + qy*qy + qz*qz;
        float qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
        float fi;
        
        // for each atom type, compute the atomic form factor f_i(q)
        for (int type = 0; type < numAtomTypes; type++) 
        {
        
            // scan through cromermann in blocks of 9 parameters
            int tind = type * 9;
            fi =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
            fi += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
            fi += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
            fi += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
            fi += cromermann[tind+8];
            
            formfactors[type] = fi; // store for use in a second
        }
        
        // accumulant
        float Qsumx;
        float Qsumy;
        Qsumx = 0;
        Qsumy = 0;
 
        // for each atom in molecule
        for( int a = 0; a < numAtoms; a++ ) 
        {
            float phase = r_x[a]*qx + r_y[a]*qy + r_z[a]*qz;
            fi = formfactors[ r_id[a] ];
            Qsumx += fi*sin(phase);
            Qsumy += fi*cos(phase);
        } // finished one molecule.
           
           
        // add the output to the total intensity array
        // #pragma omp critical
        outQ[iq] += (Qsumx*Qsumx + Qsumy*Qsumy); // / n_rotations;
        
        // discrete photon statistics will go here, if implemented 
        // we'll need a different array to accumulate the results of each
        // molecule, then add the discrete statistical draw to the final
        // output       -TJL
    } // end looping over q
}


CPUScatter::CPUScatter( int    nQ_,
                        float* h_qx_,
                        float* h_qy_,
                        float* h_qz_,
                
                        // atomic positions, ids
                        int    nAtoms_,
                        float* h_rx_,
                        float* h_ry_,
                        float* h_rz_,
                        int*   h_id_,

                        // cromer-mann parameters
                        int    nCM_,
                        float* h_cm_,

                        // output
                        float* h_outQ_ ) {
                                
    // unpack arguments
    nQ = nQ_;
    h_qx = h_qx_;
    h_qy = h_qy_;
    h_qz = h_qz_;

    nAtoms = nAtoms_;
    int numAtomTypes = nCM_ / 9;
    h_rx = h_rx_;
    h_ry = h_ry_;
    h_rz = h_rz_;
    h_id = h_id_;

    h_cm = h_cm_;

    h_outQ = h_outQ_;
    

    // execute the kernel
    kernel(h_qx, h_qy, h_qz, h_outQ, nQ, h_rx, h_ry, h_rz, h_id, nAtoms, numAtomTypes, h_cm);
}

CPUScatter::~CPUScatter() {
    // destroy the class
}
