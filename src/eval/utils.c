/*
			       utils.c

		 c  Ivo L Hofacker and Walter Fontana
			  Vienna RNA package
*/
/* Last changed Time-stamp: <2008-11-25 16:34:36 ivo> */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#include "../config.h"
#ifdef WITH_DMALLOC
#include "dmalloc.h"
#endif
/*@unused@*/
static char rcsid[] = "$Id: utils.c,v 1.19 2008/12/16 22:30:30 ivo Exp $";

#define PRIVATE  static
#define PUBLIC

/*@notnull@ @only@*/
PUBLIC void  *space(unsigned int size);
/*@exits@*/
PUBLIC void   nrerror(const char message[]);
PUBLIC double urn(void);
PUBLIC int    int_urn(int from, int to);
PUBLIC void   filecopy(FILE *from, FILE *to);
/*@observer@*/
PUBLIC char  *time_stamp(void);
PUBLIC char  *random_string(int l, const char symbols[]);
PUBLIC int    hamming(const char *s1, const char *s2);
PUBLIC char  *get_line(FILE *fp);

PUBLIC unsigned short xsubi[3];

/*-------------------------------------------------------------------------*/

PUBLIC void *space(unsigned size) {
  void *pointer;

  if ( (pointer = (void *) calloc(1, (size_t) size)) == NULL) {
#ifdef EINVAL
    if (errno==EINVAL) {
      fprintf(stderr,"SPACE: requested size: %d\n", size);
      nrerror("SPACE allocation failure -> EINVAL");
    }
    if (errno==ENOMEM)
#endif
      nrerror("SPACE allocation failure -> no memory");
  }
  return  pointer;
}

#ifdef WITH_DMALLOC
#define space(S) calloc(1,(S))
#endif

#undef xrealloc
/* dmalloc.h #define's xrealloc */
void *xrealloc (void *p, unsigned size) {
  if (p == 0)
    return space(size);
  p = (void *) realloc(p, size);
  if (p == NULL) {
#ifdef EINVAL
    if (errno==EINVAL) {
      fprintf(stderr,"xrealloc: requested size: %d\n", size);
      nrerror("xrealloc allocation failure -> EINVAL");
    }
    if (errno==ENOMEM)
#endif
      nrerror("xrealloc allocation failure -> no memory");
  }
  return p;
}

/*------------------------------------------------------------------------*/

PUBLIC void nrerror(const char message[])       /* output message upon error */
{
  fprintf(stderr, "\n%s\n", message);
  exit(EXIT_FAILURE);
}

/*------------------------------------------------------------------------*/
PUBLIC void init_rand(void)
{
  time_t t;
  (void) time(&t);
  xsubi[0] = xsubi[1] = xsubi[2] = (unsigned short) t;  /* lower 16 bit */
  xsubi[1] += (unsigned short) ((unsigned)t >> 6);
  xsubi[2] += (unsigned short) ((unsigned)t >> 12);
#ifndef HAVE_ERAND48
  srand((unsigned int) t);
#endif
}

/*------------------------------------------------------------------------*/

PUBLIC double urn(void)
     /* uniform random number generator; urn() is in [0,1] */
     /* uses a linear congruential library routine */
     /* 48 bit arithmetic */
{
#ifdef HAVE_ERAND48
  extern double erand48(unsigned short[]);
  return erand48(xsubi);
#else
  return ((double) rand())/RAND_MAX;
#endif
}

/*------------------------------------------------------------------------*/

PUBLIC int int_urn(int from, int to)
{
  return ( ( (int) (urn()*(to-from+1)) ) + from );
}

/*------------------------------------------------------------------------*/

PUBLIC void filecopy(FILE *from, FILE *to)
{
  int c;

  while ((c = getc(from)) != EOF) (void)putc(c, to);
}

/*-----------------------------------------------------------------*/

PUBLIC char *time_stamp(void)
{
  time_t  cal_time;

  cal_time = time(NULL);
  return ( ctime(&cal_time) );
}

/*-----------------------------------------------------------------*/

PUBLIC char *random_string(int l, const char symbols[])
{
  char *r;
  int   i, rn, base;

  base = (int) strlen(symbols);
  r = (char *) space(sizeof(char)*(l+1));

  for (i = 0; i < l; i++) {
    rn = (int) (urn()*base);  /* [0, base-1] */
    r[i] = symbols[rn];
  }
  r[l] = '\0';
  return r;
}

/*-----------------------------------------------------------------*/

PUBLIC int   hamming(const char *s1, const char *s2)
{
  int h=0;

  for (; *s1 && *s2; s1++, s2++)
    if (*s1 != *s2) h++;
  return h;
}
/*-----------------------------------------------------------------*/

PUBLIC char *get_line(FILE *fp) /* reads lines of arbitrary length from fp */
{
  char s[512], *line, *cp;
  int len=0, size=0, l;
  line=NULL;
  do {
    if (fgets(s, 512, fp)==NULL) break;
    cp = strchr(s, '\n');
    if (cp != NULL) *cp = '\0';
    l = len + strlen(s);
    if (l+1>size) {
      size = (l+1)*1.2;
      line = (char *) xrealloc(line, size*sizeof(char));
    }
    strcat(line+len, s);
    len=l;
  } while(cp==NULL);

  return line;
}

/*-----------------------------------------------------------------*/

PUBLIC char *pack_structure(const char *struc) {
  /* 5:1 compression using base 3 encoding */
  int i,j,l,pi;
  unsigned char *packed;

  l = (int) strlen(struc);
  packed = (unsigned char *) space(((l+4)/5+1)*sizeof(unsigned char));

  j=i=pi=0;
  while (i<l) {
    register int p;
    for (p=pi=0; pi<5; pi++) {
      p *= 3;
      switch (struc[i]) {
      case '(':
      case '\0':
	break;
      case '.':
	p++;
	break;
      case ')':
	p += 2;
	break;
      default: nrerror("pack_structure: illegal charcter in structure");
      }
      if (i<l) i++;
    }
    packed[j++] = (unsigned char) (p+1); /* never use 0, so we can use
					    strcmp()  etc. */
  }
  packed[j] = '\0';      /* for str*() functions */
  return (char *) packed;
}

PUBLIC char *unpack_structure(const char *packed) {
  /* 5:1 compression using base 3 encoding */
  int i,j,l;
  char *struc;
  unsigned const char *pp;
  char code[3] = {'(', '.', ')'};

  l = (int) strlen(packed);
  pp = (const unsigned char *) packed;
  struc = (char *) space((l*5+1)*sizeof(char));   /* up to 4 byte extra */

  for (i=j=0; i<l; i++) {
    register int p, c, k;

    p = (int) pp[i] - 1;
    for (k=4; k>=0; k--) {
      c = p % 3;
      p /= 3;
      struc[j+k] = code[c];
    }
    j += 5;
  }
  struc[j--] = '\0';
  while (struc[j] == '(') /* strip trailing ( */
    struc[j--] = '\0';

  return struc;
}

/*--------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------
   make_pair_table (Modified for Pseudoknots)
   Recognizes both () and [] to allow crossing pairs.
------------------------------------------------------------------------- */
PUBLIC short *make_pair_table(const char *structure)
{
   short i, j;
   short length;
   short *table;
   
   // スタックを2つ用意します
   short *stack_p; // Parentheses ( ) 用
   short *stack_b; // Brackets [ ] 用
   int hx_p = 0;   // Stack pointer for ( )
   int hx_b = 0;   // Stack pointer for [ ]

   length = (short) strlen(structure);
   
   // メモリ確保 (ViennaRNAのspace関数がない環境も考慮し、標準のcalloc/mallocを使用)
   // ※ 元の環境で space() が必須なら space() に書き換えてください
   table = (short *) calloc(length + 2, sizeof(short));
   stack_p = (short *) malloc(sizeof(short) * (length + 1));
   stack_b = (short *) malloc(sizeof(short) * (length + 1));

   if (!table || !stack_p || !stack_b) {
       fprintf(stderr, "Memory allocation failure in make_pair_table\n");
       exit(1);
   }

   table[0] = length;

   for (i=1; i<=length; i++) {
      switch (structure[i-1]) {
       // --- 丸括弧 ( ) の処理 ---
       case '(':
         stack_p[hx_p++] = i;
         table[i] = 0; // 一旦0で初期化
         break;
       case ')':
         if (hx_p <= 0) {
            fprintf(stderr, "%s\n", structure);
            fprintf(stderr, "unbalanced parentheses ')' at %d\n", i);
            exit(1); // nrerrorの代わり
         }
         j = stack_p[--hx_p];
         table[i] = j;
         table[j] = i;
         break;

       // --- 角括弧 [ ] の処理 (追加部分) ---
       case '[':
         stack_b[hx_b++] = i;
         table[i] = 0; // 一旦0で初期化
         break;
       case ']':
         if (hx_b <= 0) {
            fprintf(stderr, "%s\n", structure);
            fprintf(stderr, "unbalanced brackets ']' at %d\n", i);
            exit(1);
         }
         j = stack_b[--hx_b];
         table[i] = j;
         table[j] = i;
         break;

       // --- その他 ( . など) ---
       default:
         table[i] = 0;
         break;
      }
   }

   // 終了後のチェック
   if (hx_p != 0) {
      fprintf(stderr, "%s\n", structure);
      fprintf(stderr, "unbalanced parentheses '(' remaining\n");
      exit(1);
   }
   if (hx_b != 0) {
      fprintf(stderr, "%s\n", structure);
      fprintf(stderr, "unbalanced brackets '[' remaining\n");
      exit(1);
   }

   free(stack_p);
   free(stack_b);
   return(table);
}
/*---------------------------------------------------------------------------*/

PUBLIC int bp_distance(const char *str1, const char *str2)
{
  /* dist = {number of base pairs in one structure but not in the other} */
  /* same as edit distance with pair_open pair_close as move set */
   int dist;
   short i,l;
   short *t1, *t2;

   dist = 0;
   t1 = make_pair_table(str1);
   t2 = make_pair_table(str2);

   l = (t1[0]<t2[0])?t1[0]:t2[0];    /* minimum of the two lengths */

   for (i=1; i<=l; i++)
     if (t1[i]!=t2[i]) {
       if (t1[i]>i) dist++;
       if (t2[i]>i) dist++;
     }
   free(t1); free(t2);
   return dist;
}

#ifndef HAVE_STRDUP
char *strdup(const char *s) {
  char *dup;

  dup = space(strlen(s)+1);
  strcpy(dup, s);
  return(dup);
}
#endif
