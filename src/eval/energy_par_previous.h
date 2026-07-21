/* 
   prototypes for energy_par.c
*/

#include "energy_const.h"

extern double lxc37;   /* parameter for logarithmic loop
			  energy extrapolation            */

extern int stack37[NBPAIRS+1][NBPAIRS+1];
extern int enthalpies[NBPAIRS+1][NBPAIRS+1]; /* stack enthalpies */
extern int entropies[NBPAIRS+1][NBPAIRS+1];  /* not used anymore */

extern int hairpin37[31];
extern int bulge37[31];
extern int internal_loop37[31];
extern int internal2_energy;
extern int old_mismatch_37[NBPAIRS+1][5][5];
extern int mismatchI37[NBPAIRS+1][5][5];  /* interior loop mismatches */
extern int mismatchH37[NBPAIRS+1][5][5];  /* same for hairpins */
extern int mismatchM37[NBPAIRS+1][5][5];  /* same for multiloops */
extern int mism_H[NBPAIRS+1][5][5];       /* mismatch enthalpies */

extern int dangle5_37[NBPAIRS+1][5];      /* 5' dangle exterior of pair */
extern int dangle3_37[NBPAIRS+1][5];      /* 3' dangle */
extern int dangle3_H[NBPAIRS+1][5];       /* corresponding enthalpies */
extern int dangle5_H[NBPAIRS+1][5];

/* constants for linearly destabilizing contributions for multi-loops
   F = ML_closing + ML_intern*(k-1) + ML_BASE*u  */
extern int ML_BASE37;
extern int ML_closing37;
extern int ML_intern37;

/* Ninio-correction for asymmetric internal loops with branches n1 and n2 */
/*    ninio_energy = min{max_ninio, |n1-n2|*F_ninio[min{4.0, n1, n2}] } */
extern int         MAX_NINIO;                   /* maximum correction */
extern int F_ninio37[5];

/* penalty for helices terminated by AU (actually not GC) */
extern int TerminalAU;
/* penalty for forming bi-molecular duplex */
extern int DuplexInit;
/* stabilizing contribution due to special hairpins of size 4 (tetraloops) */
extern char Tetraloops[];  /* string containing the special tetraloops */
extern int  TETRA_ENERGY37[];  /* Bonus energy for special tetraloops */
extern int  TETRA_ENTH37;
extern char Triloops[];    /* string containing the special triloops */
extern int  Triloop_E37[]; /* Bonus energy for special Triloops */  

extern double Tmeasure;       /* temperature of param measurements */


enum {
    s1_min = 3, s1_max = 14,
    s2_min = 3, s2_max = 10,
    l1_min = 1, l1_max = 15,
    l2_min = 0, l2_max = 13,
    l3_min = 0, l3_max = 29,
    
    _pk_len_min = s1_min + l1_min + s2_min + l2_min + s1_min + l3_min + s2_min,
    _pk_len_max = s1_max + l1_max + s2_max + l2_max + s1_max + l3_max + s2_max
};

//===========
//dp09
//===========
//Pseudoknot initiation penalties (in 10*cal/mol)
extern const int PK_penalty_external_37;
extern const int PK_penalty_in_multi_37;
extern const int PK_penalty_in_pk_37;

// Other pseudoknot penalties (in 10*cal/mol)
extern const int PK_penalty_band_37;
extern const int PK_penalty_unpaired_37;

// nested substructure penalties
extern const int PK_penaltiy_nested;

// spans a band
// initiating a multiloop 
extern const int PK_penaltiy_inial_multi;

// branch in a multiloop 
extern const int PK_penaltiy_branch_in_multi;

// unpaired base in a multiloop 
extern const int PK_penaltiy_unpaired_base_in_multi;

// Multiplicative penalties (unitless)
extern const double PK_mult_stacked_pair;
extern const double PK_mult_internal_loop;


#include "intloops.h"