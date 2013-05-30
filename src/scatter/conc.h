
/* Header file for cpuscatter.cpp, CPUScatter class */

class CPUScatter {
    
    // declare variables
    unsigned int nQ_size;

    int nQ;
    float* h_qx;    // size: nQ
    float* h_qy;    // size: nQ
    float* h_qz;    // size: nQ

    int nAtoms;
    float* h_rx;    // size: nAtoms
    float* h_ry;    // size: nAtoms
    float* h_rz;    // size: nAtoms
    int*   h_id;
    float* h_cm;    // size: numAtomTypes*9

    float* h_outQ;  // size: nQ (OUTPUT)


public:
    CPUScatter( int    nQ_,
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
                float* h_outQ_ );
           
  ~CPUScatter();                           // destructor
};
