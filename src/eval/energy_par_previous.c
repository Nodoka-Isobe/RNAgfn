/* 
    Current free energy parameters are summarized in:

    D.H.Mathews, J. Sabina, M. ZUker, D.H. Turner
    "Expanded sequence dependence of thermodynamic parameters improves
    prediction of RNA secondary structure"
    JMB, 288, pp 911-940, 1999

    Enthalpies taken from:
    
    A. Walter, D Turner, J Kim, M Lyttle, P M"uller, D Mathews, M Zuker
    "Coaxial stckaing of helices enhances binding of oligoribonucleotides.."
    PNAS, 91, pp 9218-9222, 1994
    
    D.H. Turner, N. Sugimoto, and S.M. Freier.
    "RNA Structure Prediction",
    Ann. Rev. Biophys. Biophys. Chem. 17, 167-192, 1988.

    John A.Jaeger, Douglas H.Turner, and Michael Zuker.
    "Improved predictions of secondary structures for RNA",
    PNAS, 86, 7706-7710, October 1989.
    
    L. He, R. Kierzek, J. SantaLucia, A.E. Walter, D.H. Turner
    "Nearest-Neughbor Parameters for GU Mismatches...."
    Biochemistry 1991, 30 11124-11132

    A.E. Peritz, R. Kierzek, N, Sugimoto, D.H. Turner
    "Thermodynamic Study of Internal Loops in Oligoribonucleotides..."
    Biochemistry 1991, 30, 6428--6435

    
*/

#include "energy_const.h"
/*@unused@*/
static char rcsid[] = "$Id: energy_par.c,v 1.6 2004/08/12 12:11:57 ivo Exp $";

#define NST 0     /* Energy for nonstandard stacked pairs */
#define DEF -50   /* Default terminal mismatch, used for I */
                  /* and any non_pairing bases */
#define NSM 0     /* terminal mismatch for non standard pairs */
 
#define PUBLIC

PUBLIC double Tmeasure = 37+K0;  /* temperature of param measurements */
PUBLIC double lxc37=107.856;     /* parameter for logarithmic loop
				    energy extrapolation            */

PUBLIC int stack37[NBPAIRS+1][NBPAIRS+1] =
/*          CG     GC     GU     UG     AU     UA  */
{ {  INF,   INF,   INF,   INF,   INF,   INF,   INF, INF},
  {  INF,  -133,  -207,  -146,   -37,  -139,  -132, NST},
  {  INF,  -207,  -205,  -150,   -91,  -129,  -123, NST},
  {  INF,  -146,  -150,   -22,   -67,   -81,   -58, NST},
  {  INF,   -37,   -91,   -67,   -38,   -14,    -2, NST},
  {  INF,  -139,  -129,   -81,   -14,   -84,   -69, NST},
  {  INF,  -132,  -123,   -58,    -2,   -69,   -68, NST}};

/* enthalpies (0.01*kcal/mol at 37 C) for stacked pairs */
/* different from mfold-2.3, which uses values from mfold-2.2 */
PUBLIC int enthalpies[NBPAIRS+1][NBPAIRS+1] = 
/*          CG     GC     GU     UG     AU     UA  */
{ {  INF,   INF,   INF,   INF,   INF,   INF,   INF, INF}, 
  {  INF, -1060, -1340, -1210,  -560, -1050, -1040, NST},
  {  INF, -1340, -1490, -1260,  -830, -1140, -1240, NST},
  {  INF, -1210, -1260, -1460, -1350,  -880, -1280, NST},
  {  INF,  -560,  -830, -1350,  -930,  -320,  -700, NST},
  {  INF, -1050, -1140,  -880,  -320,  -940,  -680, NST},
  {  INF, -1040, -1240, -1280,  -700,  -680,  -770, NST},
  {  INF,   NST,   NST,   NST,   NST,   NST,   NST, NST}};


/* old values are here just for comparison */
PUBLIC int oldhairpin37[31] = { /* from ViennaRNA 1.3 */
  INF, INF, INF, 410, 490, 440, 470, 500, 510, 520, 531,
       542, 551, 560, 568, 575, 582, 589, 595, 601, 606,
       611, 616, 621, 626, 630, 634, 638, 642, 646, 650};

PUBLIC int hairpin37[31] = {
  INF, INF, INF, 364, 282, 297, 287, 259, 259, 272, 283,
  293, 303, 311, 319, 327, 334, 340, 346, 352, 358, 363,
  368, 373, 377, 382, 386, 390, 394, 398, 401,};

PUBLIC int oldbulge37[31] = {
  INF, 390, 310, 350, 420, 480, 500, 516, 531, 543, 555,
       565, 574, 583, 591, 598, 605, 612, 618, 624, 630,
       635, 640, 645, 649, 654, 658, 662, 666, 670, 673};

PUBLIC int bulge37[31] = {
  INF, 281, 157, 201, 288, 298, 272, 288, 303, 315, 327,
  337, 346, 355, 363, 370, 377, 384, 390, 396, 401, 407,
  412, 416, 421, 425, 430, 434, 438, 442, 445,};

PUBLIC int oldinternal_loop37[31] = {
  INF, INF, 410, 510, 490, 530, 570, 587, 601, 614, 625,
       635, 645, 653, 661, 669, 676, 682, 688, 694, 700,
       705, 710, 715, 720, 724, 728, 732, 736, 740, 744};

PUBLIC int internal_loop37[31] = {
  INF, INF, 83, 118, 90, 106, 121, 133, 145,
  155, 164, 173, 181, 188, 195, 202, 208, 214, 219, 225,
  230, 234, 239, 243, 248, 252, 256, 260, 263,};
  
/* terminal mismatches */
/* mismatch free energies for interior loops at 37C */
PUBLIC int mismatchI37[NBPAIRS+1][5][5] =
{
  {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},
  {
    {    0,    0,    0,    0,    0 },
    {    0,    0,    0,  -50,    0 },
    {    0,    0,    0,    0,    0 },
    {    0,  -50,    0,    0,    0 },
    {    0,    0,    0,    0,  -45 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,    0,    0,  -50,    0 },
    {    0,    0,    0,    0,    0 },
    {    0,  -50,    0,    0,    0 },
    {    0,    0,    0,    0,  -45 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   63,   63,   13,   63 },
    {    0,   63,   63,   63,   63 },
    {    0,   13,   63,   63,   63 },
    {    0,   63,   63,   63,   18 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   63,   63,   13,   63 },
    {    0,   63,   63,   63,   63 },
    {    0,   13,   63,   63,   63 },
    {    0,   63,   63,   63,   18 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   63,   63,   13,   63 },
    {    0,   63,   63,   63,   63 },
    {    0,   13,   63,   63,   63 },
    {    0,   63,   63,   63,   18 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   63,   63,   13,   63 },
    {    0,   63,   63,   63,   63 },
    {    0,   13,   63,   63,   63 },
    {    0,   63,   63,   63,   18 }
  },
};

/* mismatch free energies for hairpins at 37C */
PUBLIC int mismatchH37[NBPAIRS+1][5][5] =
{
  {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},
  {
    {    0,    0,    0,    0,    0 },
    {  -90,  -35,  -13,    3,   13 },
    {  -90,  -12,  -17,  -98,  -22 },
    {  -90,  -51,  -24,   -1,    3 },
    {  -90,  -16,  -57,  -97,  -38 },
  },
  {
    {    0,    0,    0,    0,    0 },
    {  -70,    1,  -22,  -65,  110 },
    {  -70,   11,  -27,  -17,    0 },
    {  -70,  -67,    1,  -45,   36 },
    {  -70,    8,  -39,  -55,  -92 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   60,   75,   43,  180 },
    {    0,   50,   44,   70,   -1 },
    {    0,  -92,   15,   39,  114 },
    {    0,   25,    7,   67,   36 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   43,   52,   87,  109 },
    {    0,   52,   35,  -10,   40 },
    {    0,  -26,  -28,    5,   92 },
    {    0,   42,  -38,  -14,  -30 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   41,   77,   65,  117 },
    {    0,   -2,    8,   44,   43 },
    {    0,  -31,   23,   15,   60 },
    {    0,   29,   -6,   32,  -10 }
  },
  {
    {    0,    0,    0,    0,    0 },
    {    0,   17,   52,   76,   68 },
    {    0,   53,   36,   37,   11 },
    {    0,  -39,   46,   16,   68 },
    {    0,   50,  -16,   46,   29 }
  },
};

/* mismatch energies in multiloops */
PUBLIC int mismatchM37[NBPAIRS+1][5][5];

/* these are probably junk */
/* mismatch enthalpies for temperature scaling */
PUBLIC int mism_H[NBPAIRS+1][5][5] =
{ /* no pair */
  {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},
  { /* CG */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF,-1030, -950,-1030,-1030}, /* A@  AA  AC  AG  AU */
   { DEF, -520, -450, -520, -670}, /* C@  CA  CC  CG  CU */
   { DEF, -940, -940, -940, -940}, /* G@  GA  GC  GG  GU */
   { DEF, -810, -740, -810, -860}},/* U@  UA  UC  UG  UU */
  { /* GC */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF, -520, -880, -560, -880}, /* A@  AA  AC  AG  AU */
   { DEF, -720, -310, -310, -390}, /* C@  CA  CC  CG  CU */
   { DEF, -710, -740, -620, -740}, /* G@  GA  GC  GG  GU */
   { DEF, -500, -500, -500, -570}},/* U@  UA  UC  UG  UU */
  { /* GU */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF, -430, -600, -600, -600}, /* A@  AA  AC  AG  AU */
   { DEF, -260, -240, -240, -240}, /* C@  CA  CC  CG  CU */
   { DEF, -340, -690, -690, -690}, /* G@  GA  GC  GG  GU */
   { DEF, -330, -330, -330, -330}},/* U@  UA  UC  UG  UU */
  { /* UG */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF, -720, -790, -960, -810}, /* A@  AA  AC  AG  AU */
   { DEF, -480, -480, -360, -480}, /* C@  CA  CC  CG  CU */
   { DEF, -660, -810, -920, -810}, /* G@  GA  GC  GG  GU */
   { DEF, -550, -440, -550, -360}},/* U@  UA  UC  UG  UU */
  { /* AU */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF, -430, -600, -600, -600}, /* A@  AA  AC  AG  AU */
   { DEF, -260, -240, -240, -240}, /* C@  CA  CC  CG  CU */
   { DEF, -340, -690, -690, -690}, /* G@  GA  GC  GG  GU */
   { DEF, -330, -330, -330, -330}},/* U@  UA  UC  UG  UU */
  { /* UA */
   {   0,    0,    0,    0,    0}, /* @@  @A  @C  @G  @U */
   { DEF, -400, -630, -890, -590}, /* A@  AA  AC  AG  AU */
   { DEF, -430, -510, -200, -180}, /* C@  CA  CC  CG  CU */
   { DEF, -380, -680, -890, -680}, /* G@  GA  GC  GG  GU */
   { DEF, -280, -140, -280, -140}},/* U@  UA  UC  UG  UU */
  { /* nonstandard pair */
   {DEF,DEF,DEF,DEF,DEF},{DEF,DEF,DEF,DEF,DEF},{DEF,DEF,DEF,DEF,DEF},
   {DEF,DEF,DEF,DEF,DEF},{DEF,DEF,DEF,DEF,DEF}}
};

/* 5' dangling ends (unpaird base stacks on first paired base) */
PUBLIC int dangle5_37[NBPAIRS+1][5]=
{
  {  INF,  INF,  INF,  INF,  INF },
  {    0,   -8,    0,    0,    0 },
  {    0,  -10,    0,    0,    0 },
  {    0,    0,    0,    0,    0 },
  {    0,    0,    0,    0,    0 },
  {    0,  -10,    0,    0,    0 },
  {    0,   -3,    0,    0,    0 },
  {    0,    0,    0,    0,    0 }
};

/* 3' dangling ends (unpaired base stacks on second paired base */
PUBLIC int dangle3_37[NBPAIRS+1][5]=
{
  {  INF,  INF,  INF,  INF,  INF },
  {    0,  -10,   -9,  -51,  -14 },
  {    0,  -41,    0,  -46,  -35 },
  {    0,  -10,    0,  -50,   -2 },
  {    0,  -10,  -40,  -60,    0 },
  {    0,  -10,  -13,  -25,   -7 },
  {    0,  -10,  -30,  -44,   -7 },
  {    0,    0,    0,    0,    0 }
};

/* enthalpies for temperature scaling */
PUBLIC int dangle3_H[NBPAIRS+1][5] =
{/*   @     A     C     G     U   */
   { INF,  INF,  INF,  INF,  INF},  /* no pair */
   {   0, -740, -280, -640, -360},
   {   0, -900, -410, -860, -750},
   {   0, -740, -240, -720, -490},
   {   0, -490,  -90, -550, -230},
   {   0, -570,  -70, -580, -220},
   {   0, -490,  -90, -550, -230},
   {   0,    0,    0,    0,   0}
};

PUBLIC int dangle5_H[NBPAIRS+1][5] =
{/*   @     A     C     G     U   */
   { INF,  INF,  INF,  INF,  INF},  /* no pair */
   {   0, -240,  330,   80, -140},
   {   0, -160,   70, -460,  -40},
   {   0,  160,  220,   70,  310},
   {   0, -150,  510,   10,  100},
   {   0,  160,  220,   70,  310},
   {   0,  -50,  690,  -60,  -60},
   {   0,    0,    0,    0,   0}
};


/* constants for linearly destabilizing contributions for multi-loops
   F = ML_closing + ML_intern*k + ML_BASE*u  */
/* old versions erroneously used ML_intern*(k-1) */
PUBLIC int ML_BASE37 = -2;
PUBLIC int ML_closing37 = 315;
PUBLIC int ML_intern37 =  15;

/* Ninio-correction for asymmetric internal loops with branches n1 and n2 */
/*    ninio_energy = min{max_ninio, |n1-n2|*F_ninio[min{4.0, n1, n2}] } */
PUBLIC int         MAX_NINIO = 300;                   /* maximum correction */
PUBLIC int F_ninio37 = 50;    /* only F[2] used */

/* stabilizing contribution due to special hairpins of size 4 (tetraloops) */

PUBLIC char Tetraloops[1400] =  /* place for up to 200 tetra loops */
  "GGGGAC "
  "GGUGAC "
  "CGAAAG "
  "GGAGAC "
  "CGCAAG "
  "GGAAAC "
  "CGGAAG "
  "CUUCGG "
  "CGUGAG "
  "CGAAGG "
  "CUACGG "
  "GGCAAC "
  "CGCGAG "
  "UGAGAG "
  "CGAGAG "
  "AGAAAU "
  "CGUAAG "
  "CUAACG "
  "UGAAAG "
  "GGAAGC "
  "GGGAAC "
  "UGAAAA "
  "AGCAAU "
  "AGUAAU "
  "CGGGAG "
  "AGUGAU "
  "GGCGAC "
  "GGGAGC "
  "GUGAAC "
  "UGGAAA "
;

PUBLIC int   TETRA_ENERGY37[200] = {
  -300, -300, -300, -300, -300, -300, -300, -300, -300, -250, -250, -250,
  -250, -250, -200, -200, -200, -200, -200, -150, -150, -150, -150, -150,
  -150, -150, -150, -150, -150, -150};

PUBLIC int   TETRA_ENTH37   = -400;

PUBLIC char Triloops[241] = "";

PUBLIC int Triloop_E37[40];

/* penalty for AU (or GU) terminating helix) */
/* mismatches already contain these */
PUBLIC int TerminalAU = 56;

/* penalty for forming a bi-molecular duplex */
PUBLIC int DuplexInit = 410;
// =======================================================================
// Pseudoknot parameters from Andronescu et al. (2010) RNA, Table 6 (dp09)
// =======================================================================

//===========
//dp09
//===========
//Pseudoknot initiation penalties (in 10*cal/mol)
PUBLIC int PK_penalty_external_37 = -138;     // -1.38 kcal/mol
PUBLIC int PK_penalty_in_multi_37 = 1007;   // 10.07 kcal/mol 
PUBLIC int PK_penalty_in_pk_37 = 1500;      // 15 kcal/mol

// Other pseudoknot penalties (in 10*cal/mol)
PUBLIC int PK_penalty_band_37 = 246;        //  2.46 kcal/mol
PUBLIC int PK_penalty_unpaired_37 = 6;      //  0.06 kcal/mol

// nested substructure penalties
PUBLIC int PK_penaltiy_nested = 96;

// spans a band
// initiating a multiloop 
PUBLIC int PK_penaltiy_inial_multi = 341;

// branch in a multiloop 
PUBLIC int PK_penaltiy_branch_in_multi = 56;

// unpaired base in a multiloop 
PUBLIC int PK_penaltiy_unpaired_base_in_multi = 12;

// Multiplicative penalties (unitless)
PUBLIC double PK_mult_stacked_pair = 89;
PUBLIC double PK_mult_internal_loop = 74;


#include "intloops.h"
