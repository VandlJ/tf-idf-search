
/* This file was generated automatically by the Snowball to ANSI C compiler */

#include "../runtime/header.h"

#ifdef __cplusplus
extern "C" {
#endif
extern int czech_UTF_8_stem(struct SN_env * z);
#ifdef __cplusplus
}
#endif
static int r_adverbs_suffixes(struct SN_env * z);
static int r_verbs_endings(struct SN_env * z);
static int r_verbs_suffixes(struct SN_env * z);
static int r_adjectives_endings(struct SN_env * z);
static int r_adjectives_suffixes(struct SN_env * z);
static int r_nouns_endings(struct SN_env * z);
static int r_nouns_suffixes(struct SN_env * z);
static int r_exception(struct SN_env * z);
static int r_alternance_i(struct SN_env * z);
static int r_end_double(struct SN_env * z);
static int r_end_vowel(struct SN_env * z);
static int r_R1(struct SN_env * z);
static int r_mark_regions(struct SN_env * z);
static int r_postlude(struct SN_env * z);
#ifdef __cplusplus
extern "C" {
#endif


extern struct SN_env * czech_UTF_8_create_env(void);
extern void czech_UTF_8_close_env(struct SN_env * z);


#ifdef __cplusplus
}
#endif
static const symbol s_0_1[2] = { 0xC3, 0xA1 };
static const symbol s_0_2[2] = { 0xC3, 0xA9 };
static const symbol s_0_3[2] = { 0xC3, 0xAD };
static const symbol s_0_4[2] = { 0xC3, 0xB3 };
static const symbol s_0_5[2] = { 0xC3, 0xBA };
static const symbol s_0_6[2] = { 0xC3, 0xBD };
static const symbol s_0_7[2] = { 0xC4, 0x8D };
static const symbol s_0_8[2] = { 0xC4, 0x8F };
static const symbol s_0_9[2] = { 0xC4, 0x9B };
static const symbol s_0_10[2] = { 0xC5, 0x88 };
static const symbol s_0_11[2] = { 0xC5, 0x99 };
static const symbol s_0_12[2] = { 0xC5, 0xA1 };
static const symbol s_0_13[2] = { 0xC5, 0xA5 };
static const symbol s_0_14[2] = { 0xC5, 0xAF };
static const symbol s_0_15[2] = { 0xC5, 0xBE };

static const struct among a_0[16] =
{
/*  0 */ { 0, 0, -1, 16, 0},
/*  1 */ { 2, s_0_1, 0, 1, 0},
/*  2 */ { 2, s_0_2, 0, 2, 0},
/*  3 */ { 2, s_0_3, 0, 3, 0},
/*  4 */ { 2, s_0_4, 0, 4, 0},
/*  5 */ { 2, s_0_5, 0, 5, 0},
/*  6 */ { 2, s_0_6, 0, 6, 0},
/*  7 */ { 2, s_0_7, 0, 7, 0},
/*  8 */ { 2, s_0_8, 0, 8, 0},
/*  9 */ { 2, s_0_9, 0, 9, 0},
/* 10 */ { 2, s_0_10, 0, 10, 0},
/* 11 */ { 2, s_0_11, 0, 11, 0},
/* 12 */ { 2, s_0_12, 0, 12, 0},
/* 13 */ { 2, s_0_13, 0, 13, 0},
/* 14 */ { 2, s_0_14, 0, 14, 0},
/* 15 */ { 2, s_0_15, 0, 15, 0}
};

static const symbol s_1_0[4] = { 'b', 'u', 'd', 'e' };
static const symbol s_1_1[6] = { 'b', 'u', 'd', 'e', 'm', 'e' };
static const symbol s_1_2[6] = { 'b', 'u', 'd', 'e', 't', 'e' };
static const symbol s_1_3[6] = { 'b', 'u', 'd', 'e', 0xC5, 0xA1 };
static const symbol s_1_4[5] = { 'b', 'u', 'd', 'o', 'u' };
static const symbol s_1_5[4] = { 'b', 'u', 'd', 'u' };
static const symbol s_1_6[4] = { 'b', 'u', 0xC4, 0x8F };
static const symbol s_1_7[6] = { 'b', 'u', 0xC4, 0x8F, 'm', 'e' };
static const symbol s_1_8[6] = { 'b', 'u', 0xC4, 0x8F, 't', 'e' };
static const symbol s_1_9[3] = { 'b', 'y', 'l' };
static const symbol s_1_10[4] = { 'c', 'h', 'c', 'e' };
static const symbol s_1_11[6] = { 'c', 'h', 'c', 'e', 'm', 'e' };
static const symbol s_1_12[6] = { 'c', 'h', 'c', 'e', 't', 'e' };
static const symbol s_1_13[6] = { 'c', 'h', 'c', 'e', 0xC5, 0xA1 };
static const symbol s_1_14[4] = { 'c', 'h', 'c', 'i' };
static const symbol s_1_15[6] = { 'c', 'h', 't', 0xC3, 0xAD, 'c' };
static const symbol s_1_16[7] = { 'c', 'h', 't', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_1_17[5] = { 'c', 'h', 't', 0xC4, 0x9B };
static const symbol s_1_18[6] = { 'c', 'h', 't', 0xC4, 0x9B, 'j' };
static const symbol s_1_19[7] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 'e' };
static const symbol s_1_20[8] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 'm', 'e' };
static const symbol s_1_21[8] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 't', 'e' };
static const symbol s_1_22[8] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 0xC3, 0xAD };
static const symbol s_1_23[9] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 0xC3, 0xAD, 'c' };
static const symbol s_1_24[10] = { 'c', 'h', 't', 0xC4, 0x9B, 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_1_25[6] = { 'c', 'h', 't', 0xC4, 0x9B, 'l' };
static const symbol s_1_26[6] = { 'c', 'h', 't', 0xC4, 0x9B, 'n' };
static const symbol s_1_27[3] = { 'j', 'd', 'a' };
static const symbol s_1_28[3] = { 'j', 'd', 'e' };
static const symbol s_1_29[5] = { 'j', 'd', 'e', 'm', 'e' };
static const symbol s_1_30[5] = { 'j', 'd', 'e', 't', 'e' };
static const symbol s_1_31[5] = { 'j', 'd', 'e', 0xC5, 0xA1 };
static const symbol s_1_32[3] = { 'j', 'd', 'i' };
static const symbol s_1_33[4] = { 'j', 'd', 'o', 'u' };
static const symbol s_1_34[5] = { 'j', 'd', 'o', 'u', 'c' };
static const symbol s_1_35[6] = { 'j', 'd', 'o', 'u', 'c', 'e' };
static const symbol s_1_36[3] = { 'j', 'd', 'u' };
static const symbol s_1_37[6] = { 'j', 'd', 0xC4, 0x9B, 'm', 'e' };
static const symbol s_1_38[6] = { 'j', 'd', 0xC4, 0x9B, 't', 'e' };
static const symbol s_1_39[2] = { 'j', 'e' };
static const symbol s_1_40[4] = { 'j', 'e', 'd', 'a' };
static const symbol s_1_41[5] = { 'j', 'e', 'd', 'e', 'n' };
static const symbol s_1_42[4] = { 'j', 'e', 'd', 'l' };
static const symbol s_1_43[6] = { 'j', 'e', 'd', 'o', 'u', 'c' };
static const symbol s_1_44[7] = { 'j', 'e', 'd', 'o', 'u', 'c', 'e' };
static const symbol s_1_45[5] = { 'j', 'e', 'd', 0xC3, 0xAD };
static const symbol s_1_46[3] = { 'j', 'e', 'z' };
static const symbol s_1_47[5] = { 'j', 'e', 'z', 'm', 'e' };
static const symbol s_1_48[5] = { 'j', 'e', 'z', 't', 'e' };
static const symbol s_1_49[3] = { 'j', 's', 'a' };
static const symbol s_1_50[4] = { 'j', 's', 'e', 'm' };
static const symbol s_1_51[3] = { 'j', 's', 'i' };
static const symbol s_1_52[4] = { 'j', 's', 'm', 'e' };
static const symbol s_1_53[4] = { 'j', 's', 'o', 'u' };
static const symbol s_1_54[5] = { 'j', 's', 'o', 'u', 'c' };
static const symbol s_1_55[6] = { 'j', 's', 'o', 'u', 'c', 'e' };
static const symbol s_1_56[4] = { 'j', 's', 't', 'e' };
static const symbol s_1_57[3] = { 'j', 0xC3, 0xAD };
static const symbol s_1_58[4] = { 'j', 0xC3, 0xAD, 'm' };
static const symbol s_1_59[5] = { 'j', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_1_60[5] = { 'j', 0xC3, 0xAD, 't', 'e' };
static const symbol s_1_61[5] = { 'j', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_1_62[4] = { 'm', 'a', 'j', 'e' };
static const symbol s_1_63[5] = { 'm', 'a', 'j', 0xC3, 0xAD };
static const symbol s_1_64[6] = { 'm', 'a', 'j', 0xC3, 0xAD, 'c' };
static const symbol s_1_65[7] = { 'm', 'a', 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_1_66[3] = { 'm', 0xC3, 0xA1 };
static const symbol s_1_67[4] = { 'm', 0xC3, 0xA1, 'm' };
static const symbol s_1_68[5] = { 'm', 0xC3, 0xA1, 'm', 'e' };
static const symbol s_1_69[5] = { 'm', 0xC3, 0xA1, 't', 'e' };
static const symbol s_1_70[5] = { 'm', 0xC3, 0xA1, 0xC5, 0xA1 };
static const symbol s_1_71[4] = { 'm', 0xC4, 0x9B, 'j' };
static const symbol s_1_72[6] = { 'm', 0xC4, 0x9B, 'j', 'm', 'e' };
static const symbol s_1_73[6] = { 'm', 0xC4, 0x9B, 'j', 't', 'e' };
static const symbol s_1_74[4] = { 'm', 0xC4, 0x9B, 'l' };
static const symbol s_1_75[5] = { 'p', 'o', 'j', 0xC4, 0x8F };
static const symbol s_1_76[7] = { 'p', 'o', 'j', 0xC4, 0x8F, 't', 'e' };
static const symbol s_1_77[6] = { 'p', 0xC5, 0xAF, 'j', 'd', 'e' };
static const symbol s_1_78[8] = { 'p', 0xC5, 0xAF, 'j', 'd', 'e', 'm', 'e' };
static const symbol s_1_79[8] = { 'p', 0xC5, 0xAF, 'j', 'd', 'e', 't', 'e' };
static const symbol s_1_80[8] = { 'p', 0xC5, 0xAF, 'j', 'd', 'e', 0xC5, 0xA1 };
static const symbol s_1_81[7] = { 'p', 0xC5, 0xAF, 'j', 'd', 'o', 'u' };
static const symbol s_1_82[6] = { 'p', 0xC5, 0xAF, 'j', 'd', 'u' };
static const symbol s_1_83[4] = { 's', 'n', 0xC3, 0xAD };
static const symbol s_1_84[5] = { 's', 'n', 0xC3, 0xAD, 'm' };
static const symbol s_1_85[6] = { 's', 'n', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_1_86[6] = { 's', 'n', 0xC3, 0xAD, 't', 'e' };
static const symbol s_1_87[6] = { 's', 'n', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_1_88[7] = { 's', 'n', 0xC4, 0x9B, 'd', 'e', 'n' };
static const symbol s_1_89[6] = { 's', 'n', 0xC4, 0x9B, 'd', 'l' };
static const symbol s_1_90[7] = { 's', 'n', 0xC4, 0x9B, 'd', 0xC3, 0xAD };
static const symbol s_1_91[5] = { 's', 'n', 0xC4, 0x9B, 'z' };
static const symbol s_1_92[7] = { 's', 'n', 0xC4, 0x9B, 'z', 'm', 'e' };
static const symbol s_1_93[7] = { 's', 'n', 0xC4, 0x9B, 'z', 't', 'e' };
static const symbol s_1_94[4] = { 'v', 'i', 'd', 'a' };
static const symbol s_1_95[6] = { 'v', 'i', 'd', 'o', 'u', 'c' };
static const symbol s_1_96[7] = { 'v', 'i', 'd', 'o', 'u', 'c', 'e' };
static const symbol s_1_97[5] = { 'v', 'i', 'd', 0xC3, 0xAD };
static const symbol s_1_98[6] = { 'v', 'i', 'd', 0xC3, 0xAD, 'm' };
static const symbol s_1_99[7] = { 'v', 'i', 'd', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_1_100[7] = { 'v', 'i', 'd', 0xC3, 0xAD, 't', 'e' };
static const symbol s_1_101[7] = { 'v', 'i', 'd', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_1_102[6] = { 'v', 'i', 'd', 0xC4, 0x9B, 'l' };
static const symbol s_1_103[6] = { 'v', 'i', 'd', 0xC4, 0x9B, 'n' };
static const symbol s_1_104[3] = { 'v', 'i', 'z' };
static const symbol s_1_105[5] = { 'v', 'i', 'z', 'm', 'e' };
static const symbol s_1_106[5] = { 'v', 'i', 'z', 't', 'e' };
static const symbol s_1_107[3] = { 'v', 0xC3, 0xAD };
static const symbol s_1_108[4] = { 'v', 0xC3, 0xAD, 'm' };
static const symbol s_1_109[5] = { 'v', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_1_110[5] = { 'v', 0xC3, 0xAD, 't', 'e' };
static const symbol s_1_111[5] = { 'v', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_1_112[5] = { 'v', 0xC4, 0x9B, 'd', 'a' };
static const symbol s_1_113[7] = { 'v', 0xC4, 0x9B, 'd', 'o', 'u', 'c' };
static const symbol s_1_114[8] = { 'v', 0xC4, 0x9B, 'd', 'o', 'u', 'c', 'e' };
static const symbol s_1_115[6] = { 'v', 0xC4, 0x9B, 'd', 0xC3, 0xAD };
static const symbol s_1_116[7] = { 'v', 0xC4, 0x9B, 'd', 0xC4, 0x9B, 'l' };
static const symbol s_1_117[7] = { 'v', 0xC4, 0x9B, 'd', 0xC4, 0x9B, 'n' };
static const symbol s_1_118[4] = { 'v', 0xC4, 0x9B, 'z' };
static const symbol s_1_119[6] = { 'v', 0xC4, 0x9B, 'z', 'm', 'e' };
static const symbol s_1_120[6] = { 'v', 0xC4, 0x9B, 'z', 't', 'e' };
static const symbol s_1_121[4] = { 0xC5, 0xA1, 'e', 'l' };
static const symbol s_1_122[4] = { 0xC5, 0xA1, 'l', 'a' };
static const symbol s_1_123[4] = { 0xC5, 0xA1, 'l', 'i' };
static const symbol s_1_124[4] = { 0xC5, 0xA1, 'l', 'o' };

static const struct among a_1[125] =
{
/*  0 */ { 4, s_1_0, -1, 1, 0},
/*  1 */ { 6, s_1_1, 0, 1, 0},
/*  2 */ { 6, s_1_2, 0, 1, 0},
/*  3 */ { 6, s_1_3, 0, 1, 0},
/*  4 */ { 5, s_1_4, -1, 1, 0},
/*  5 */ { 4, s_1_5, -1, 1, 0},
/*  6 */ { 4, s_1_6, -1, 1, 0},
/*  7 */ { 6, s_1_7, 6, 1, 0},
/*  8 */ { 6, s_1_8, 6, 1, 0},
/*  9 */ { 3, s_1_9, -1, 1, 0},
/* 10 */ { 4, s_1_10, -1, 8, 0},
/* 11 */ { 6, s_1_11, 10, 8, 0},
/* 12 */ { 6, s_1_12, 10, 8, 0},
/* 13 */ { 6, s_1_13, 10, 8, 0},
/* 14 */ { 4, s_1_14, -1, 8, 0},
/* 15 */ { 6, s_1_15, -1, 8, 0},
/* 16 */ { 7, s_1_16, 15, 8, 0},
/* 17 */ { 5, s_1_17, -1, 8, 0},
/* 18 */ { 6, s_1_18, 17, 8, 0},
/* 19 */ { 7, s_1_19, 18, 8, 0},
/* 20 */ { 8, s_1_20, 18, 8, 0},
/* 21 */ { 8, s_1_21, 18, 8, 0},
/* 22 */ { 8, s_1_22, 18, 8, 0},
/* 23 */ { 9, s_1_23, 22, 8, 0},
/* 24 */ { 10, s_1_24, 23, 8, 0},
/* 25 */ { 6, s_1_25, 17, 8, 0},
/* 26 */ { 6, s_1_26, 17, 8, 0},
/* 27 */ { 3, s_1_27, -1, 3, 0},
/* 28 */ { 3, s_1_28, -1, 3, 0},
/* 29 */ { 5, s_1_29, 28, 3, 0},
/* 30 */ { 5, s_1_30, 28, 3, 0},
/* 31 */ { 5, s_1_31, 28, 3, 0},
/* 32 */ { 3, s_1_32, -1, 3, 0},
/* 33 */ { 4, s_1_33, -1, 3, 0},
/* 34 */ { 5, s_1_34, 33, 3, 0},
/* 35 */ { 6, s_1_35, 34, 3, 0},
/* 36 */ { 3, s_1_36, -1, 3, 0},
/* 37 */ { 6, s_1_37, -1, 3, 0},
/* 38 */ { 6, s_1_38, -1, 3, 0},
/* 39 */ { 2, s_1_39, -1, 1, 0},
/* 40 */ { 4, s_1_40, 39, 4, 0},
/* 41 */ { 5, s_1_41, 39, 4, 0},
/* 42 */ { 4, s_1_42, 39, 4, 0},
/* 43 */ { 6, s_1_43, 39, 4, 0},
/* 44 */ { 7, s_1_44, 43, 4, 0},
/* 45 */ { 5, s_1_45, 39, 4, 0},
/* 46 */ { 3, s_1_46, 39, 4, 0},
/* 47 */ { 5, s_1_47, 46, 4, 0},
/* 48 */ { 5, s_1_48, 46, 4, 0},
/* 49 */ { 3, s_1_49, -1, 1, 0},
/* 50 */ { 4, s_1_50, -1, 1, 0},
/* 51 */ { 3, s_1_51, -1, 1, 0},
/* 52 */ { 4, s_1_52, -1, 1, 0},
/* 53 */ { 4, s_1_53, -1, 1, 0},
/* 54 */ { 5, s_1_54, 53, 1, 0},
/* 55 */ { 6, s_1_55, 54, 1, 0},
/* 56 */ { 4, s_1_56, -1, 1, 0},
/* 57 */ { 3, s_1_57, -1, 4, 0},
/* 58 */ { 4, s_1_58, 57, 4, 0},
/* 59 */ { 5, s_1_59, 58, 4, 0},
/* 60 */ { 5, s_1_60, 57, 4, 0},
/* 61 */ { 5, s_1_61, 57, 4, 0},
/* 62 */ { 4, s_1_62, -1, 2, 0},
/* 63 */ { 5, s_1_63, -1, 2, 0},
/* 64 */ { 6, s_1_64, 63, 2, 0},
/* 65 */ { 7, s_1_65, 64, 2, 0},
/* 66 */ { 3, s_1_66, -1, 2, 0},
/* 67 */ { 4, s_1_67, 66, 2, 0},
/* 68 */ { 5, s_1_68, 67, 2, 0},
/* 69 */ { 5, s_1_69, 66, 2, 0},
/* 70 */ { 5, s_1_70, 66, 2, 0},
/* 71 */ { 4, s_1_71, -1, 2, 0},
/* 72 */ { 6, s_1_72, 71, 2, 0},
/* 73 */ { 6, s_1_73, 71, 2, 0},
/* 74 */ { 4, s_1_74, -1, 2, 0},
/* 75 */ { 5, s_1_75, -1, 3, 0},
/* 76 */ { 7, s_1_76, 75, 3, 0},
/* 77 */ { 6, s_1_77, -1, 3, 0},
/* 78 */ { 8, s_1_78, 77, 3, 0},
/* 79 */ { 8, s_1_79, 77, 3, 0},
/* 80 */ { 8, s_1_80, 77, 3, 0},
/* 81 */ { 7, s_1_81, -1, 3, 0},
/* 82 */ { 6, s_1_82, -1, 3, 0},
/* 83 */ { 4, s_1_83, -1, 5, 0},
/* 84 */ { 5, s_1_84, 83, 5, 0},
/* 85 */ { 6, s_1_85, 84, 5, 0},
/* 86 */ { 6, s_1_86, 83, 5, 0},
/* 87 */ { 6, s_1_87, 83, 5, 0},
/* 88 */ { 7, s_1_88, -1, 5, 0},
/* 89 */ { 6, s_1_89, -1, 5, 0},
/* 90 */ { 7, s_1_90, -1, 5, 0},
/* 91 */ { 5, s_1_91, -1, 5, 0},
/* 92 */ { 7, s_1_92, 91, 5, 0},
/* 93 */ { 7, s_1_93, 91, 5, 0},
/* 94 */ { 4, s_1_94, -1, 6, 0},
/* 95 */ { 6, s_1_95, -1, 6, 0},
/* 96 */ { 7, s_1_96, 95, 6, 0},
/* 97 */ { 5, s_1_97, -1, 6, 0},
/* 98 */ { 6, s_1_98, 97, 6, 0},
/* 99 */ { 7, s_1_99, 98, 6, 0},
/*100 */ { 7, s_1_100, 97, 6, 0},
/*101 */ { 7, s_1_101, 97, 6, 0},
/*102 */ { 6, s_1_102, -1, 6, 0},
/*103 */ { 6, s_1_103, -1, 6, 0},
/*104 */ { 3, s_1_104, -1, 6, 0},
/*105 */ { 5, s_1_105, 104, 6, 0},
/*106 */ { 5, s_1_106, 104, 6, 0},
/*107 */ { 3, s_1_107, -1, 7, 0},
/*108 */ { 4, s_1_108, 107, 7, 0},
/*109 */ { 5, s_1_109, 108, 7, 0},
/*110 */ { 5, s_1_110, 107, 7, 0},
/*111 */ { 5, s_1_111, 107, 7, 0},
/*112 */ { 5, s_1_112, -1, 7, 0},
/*113 */ { 7, s_1_113, -1, 7, 0},
/*114 */ { 8, s_1_114, 113, 7, 0},
/*115 */ { 6, s_1_115, -1, 7, 0},
/*116 */ { 7, s_1_116, -1, 7, 0},
/*117 */ { 7, s_1_117, -1, 7, 0},
/*118 */ { 4, s_1_118, -1, 7, 0},
/*119 */ { 6, s_1_119, 118, 7, 0},
/*120 */ { 6, s_1_120, 118, 7, 0},
/*121 */ { 4, s_1_121, -1, 3, 0},
/*122 */ { 4, s_1_122, -1, 3, 0},
/*123 */ { 4, s_1_123, -1, 3, 0},
/*124 */ { 4, s_1_124, -1, 3, 0}
};

static const symbol s_2_0[2] = { 'b', 'a' };
static const symbol s_2_1[3] = { 'o', 'b', 'a' };
static const symbol s_2_2[3] = { 't', 'b', 'a' };
static const symbol s_2_3[4] = { 'i', 't', 'b', 'a' };
static const symbol s_2_4[3] = { 'n', 'd', 'a' };
static const symbol s_2_5[4] = { 'u', 'c', 'h', 'a' };
static const symbol s_2_6[2] = { 'k', 'a' };
static const symbol s_2_7[3] = { 'i', 'k', 'a' };
static const symbol s_2_8[3] = { 'n', 'k', 'a' };
static const symbol s_2_9[4] = { 'e', 'n', 'k', 'a' };
static const symbol s_2_10[4] = { 'i', 'n', 'k', 'a' };
static const symbol s_2_11[5] = { 0xC4, 0x9B, 'n', 'k', 'a' };
static const symbol s_2_12[3] = { 't', 'k', 'a' };
static const symbol s_2_13[3] = { 'v', 'k', 'a' };
static const symbol s_2_14[4] = { 0xC4, 0x8D, 'k', 'a' };
static const symbol s_2_15[5] = { 'e', 0xC4, 0x8D, 'k', 'a' };
static const symbol s_2_16[6] = { 0xC3, 0xAD, 0xC4, 0x8D, 'k', 'a' };
static const symbol s_2_17[5] = { 'u', 0xC5, 0xA1, 'k', 'a' };
static const symbol s_2_18[5] = { 't', 0xC3, 0xA9, 'k', 'a' };
static const symbol s_2_19[3] = { 'a', 'l', 'a' };
static const symbol s_2_20[4] = { 0xC3, 0xA1, 'l', 'a' };
static const symbol s_2_21[2] = { 'n', 'a' };
static const symbol s_2_22[3] = { 'i', 'n', 'a' };
static const symbol s_2_23[4] = { 't', 'i', 'n', 'a' };
static const symbol s_2_24[6] = { 0xC5, 0xA1, 't', 'i', 'n', 'a' };
static const symbol s_2_25[5] = { 'o', 'v', 'i', 'n', 'a' };
static const symbol s_2_26[5] = { 't', 'e', 'l', 'n', 'a' };
static const symbol s_2_27[5] = { 0xC3, 0xA1, 'r', 'n', 'a' };
static const symbol s_2_28[5] = { 0xC3, 0xAD, 'r', 'n', 'a' };
static const symbol s_2_29[4] = { 'o', 'v', 'n', 'a' };
static const symbol s_2_30[4] = { 'i', 'z', 'n', 'a' };
static const symbol s_2_31[5] = { 'a', 'j', 'z', 'n', 'a' };
static const symbol s_2_32[3] = { 'u', 'r', 'a' };
static const symbol s_2_33[4] = { 't', 'u', 'r', 'a' };
static const symbol s_2_34[3] = { 'i', 't', 'a' };
static const symbol s_2_35[3] = { 'o', 't', 'a' };
static const symbol s_2_36[4] = { 'i', 's', 't', 'a' };
static const symbol s_2_37[3] = { 't', 'v', 'a' };
static const symbol s_2_38[2] = { 'e', 'c' };
static const symbol s_2_39[4] = { 'i', 'n', 'e', 'c' };
static const symbol s_2_40[6] = { 'o', 'b', 'i', 'n', 'e', 'c' };
static const symbol s_2_41[3] = { 't', 'e', 'c' };
static const symbol s_2_42[4] = { 'o', 'v', 'e', 'c' };
static const symbol s_2_43[4] = { 0xC5, 0x99, 'a', 'd' };
static const symbol s_2_44[3] = { 'v', 'o', 'd' };
static const symbol s_2_45[2] = { 'c', 'e' };
static const symbol s_2_46[3] = { 'a', 'c', 'e' };
static const symbol s_2_47[3] = { 'i', 'c', 'e' };
static const symbol s_2_48[4] = { 'n', 'i', 'c', 'e' };
static const symbol s_2_49[5] = { 'e', 'n', 'i', 'c', 'e' };
static const symbol s_2_50[3] = { 'n', 'c', 'e' };
static const symbol s_2_51[4] = { 'a', 'n', 'c', 'e' };
static const symbol s_2_52[4] = { 'e', 'n', 'c', 'e' };
static const symbol s_2_53[2] = { 'i', 'e' };
static const symbol s_2_54[4] = { 'e', 'r', 'i', 'e' };
static const symbol s_2_55[3] = { 'u', 'l', 'e' };
static const symbol s_2_56[2] = { 's', 'e' };
static const symbol s_2_57[2] = { 'x', 'e' };
static const symbol s_2_58[2] = { 'z', 'e' };
static const symbol s_2_59[3] = { 0xC4, 0x8D, 'e' };
static const symbol s_2_60[4] = { 'g', 'r', 'a', 'f' };
static const symbol s_2_61[3] = { 'i', 'n', 'g' };
static const symbol s_2_62[3] = { 'l', 'o', 'g' };
static const symbol s_2_63[3] = { 'o', 'c', 'h' };
static const symbol s_2_64[2] = { 'e', 'k' };
static const symbol s_2_65[5] = { 0xC3, 0xA1, 'n', 'e', 'k' };
static const symbol s_2_66[5] = { 0xC3, 0xAD, 'n', 'e', 'k' };
static const symbol s_2_67[5] = { 0xC3, 0xA1, 's', 'e', 'k' };
static const symbol s_2_68[5] = { 'e', 0xC4, 0x8D, 'e', 'k' };
static const symbol s_2_69[6] = { 0xC3, 0xA1, 0xC4, 0x8D, 'e', 'k' };
static const symbol s_2_70[6] = { 0xC3, 0xAD, 0xC4, 0x8D, 'e', 'k' };
static const symbol s_2_71[6] = { 'o', 'u', 0xC5, 0xA1, 'e', 'k' };
static const symbol s_2_72[2] = { 'i', 'k' };
static const symbol s_2_73[3] = { 'i', 'n', 'k' };
static const symbol s_2_74[3] = { 'u', 'n', 'k' };
static const symbol s_2_75[4] = { 'u', 0xC5, 0x88, 'k' };
static const symbol s_2_76[3] = { 0xC3, 0xA1, 'k' };
static const symbol s_2_77[3] = { 0xC3, 0xAD, 'k' };
static const symbol s_2_78[4] = { 'n', 0xC3, 0xAD, 'k' };
static const symbol s_2_79[6] = { 'o', 'v', 'n', 0xC3, 0xAD, 'k' };
static const symbol s_2_80[3] = { 't', 'e', 'l' };
static const symbol s_2_81[6] = { 'o', 'v', 'a', 't', 'e', 'l' };
static const symbol s_2_82[4] = { 'i', 't', 'e', 'l' };
static const symbol s_2_83[3] = { 0xC3, 0xA1, 'l' };
static const symbol s_2_84[3] = { 'n', 'o', 'm' };
static const symbol s_2_85[3] = { 'i', 'u', 'm' };
static const symbol s_2_86[4] = { 'i', 'k', 'u', 'm' };
static const symbol s_2_87[4] = { 'i', 'v', 'u', 'm' };
static const symbol s_2_88[2] = { 'a', 'n' };
static const symbol s_2_89[4] = { 0xC4, 0x8D, 'a', 'n' };
static const symbol s_2_90[3] = { 'o', 'u', 'n' };
static const symbol s_2_91[4] = { 'l', 'o', 'u', 'n' };
static const symbol s_2_92[3] = { 0xC3, 0xA1, 'n' };
static const symbol s_2_93[4] = { 'i', 0xC3, 0xA1, 'n' };
static const symbol s_2_94[3] = { 0xC3, 0xAD, 'n' };
static const symbol s_2_95[2] = { 'k', 'o' };
static const symbol s_2_96[4] = { 'i', 's', 'k', 'o' };
static const symbol s_2_97[6] = { 'o', 'v', 'i', 's', 'k', 'o' };
static const symbol s_2_98[3] = { 't', 'k', 'o' };
static const symbol s_2_99[5] = { 0xC3, 0xA1, 't', 'k', 'o' };
static const symbol s_2_100[5] = { 'e', 0xC4, 0x8D, 'k', 'o' };
static const symbol s_2_101[6] = { 0xC3, 0xA9, 0xC4, 0x8D, 'k', 'o' };
static const symbol s_2_102[6] = { 0xC3, 0xAD, 0xC4, 0x8D, 'k', 'o' };
static const symbol s_2_103[2] = { 'l', 'o' };
static const symbol s_2_104[3] = { 'd', 'l', 'o' };
static const symbol s_2_105[2] = { 'n', 'o' };
static const symbol s_2_106[4] = { 0xC4, 0x8D, 'n', 'o' };
static const symbol s_2_107[3] = { 'i', 'v', 'o' };
static const symbol s_2_108[3] = { 't', 'v', 'o' };
static const symbol s_2_109[4] = { 's', 't', 'v', 'o' };
static const symbol s_2_110[6] = { 'o', 'v', 's', 't', 'v', 'o' };
static const symbol s_2_111[2] = { 'e', 'r' };
static const symbol s_2_112[2] = { 'o', 'r' };
static const symbol s_2_113[3] = { 't', 'o', 'r' };
static const symbol s_2_114[5] = { 0xC3, 0xA1, 't', 'o', 'r' };
static const symbol s_2_115[3] = { 'o', 'u', 'r' };
static const symbol s_2_116[4] = { 'm', 0xC4, 0x9B, 'r' };
static const symbol s_2_117[3] = { 0xC3, 0xA9, 'r' };
static const symbol s_2_118[4] = { 'i', 0xC3, 0xA9, 'r' };
static const symbol s_2_119[5] = { 'a', 't', 0xC3, 0xA9, 'r' };
static const symbol s_2_120[2] = { 'a', 's' };
static const symbol s_2_121[5] = { 'i', 's', 'm', 'u', 's' };
static const symbol s_2_122[3] = { 'm', 'e', 't' };
static const symbol s_2_123[3] = { 'a', 'n', 't' };
static const symbol s_2_124[3] = { 'e', 'n', 't' };
static const symbol s_2_125[4] = { 'm', 'e', 'n', 't' };
static const symbol s_2_126[2] = { 'o', 't' };
static const symbol s_2_127[3] = { 'e', 's', 't' };
static const symbol s_2_128[3] = { 'o', 's', 't' };
static const symbol s_2_129[4] = { 'n', 'o', 's', 't' };
static const symbol s_2_130[3] = { 'o', 'u', 't' };
static const symbol s_2_131[3] = { 0xC3, 0xA1, 't' };
static const symbol s_2_132[2] = { 'o', 'v' };
static const symbol s_2_133[3] = { 'i', 'n', 'y' };
static const symbol s_2_134[3] = { 'o', 0xC5, 0x88 };
static const symbol s_2_135[4] = { 0xC3, 0xA1, 0xC4, 0x8D };
static const symbol s_2_136[3] = { 'a', 0xC5, 0x99 };
static const symbol s_2_137[4] = { 't', 'o', 0xC5, 0x99 };
static const symbol s_2_138[4] = { 0xC3, 0xA1, 0xC5, 0x99 };
static const symbol s_2_139[5] = { 'k', 0xC3, 0xA1, 0xC5, 0x99 };
static const symbol s_2_140[7] = { 'i', 'o', 'n', 0xC3, 0xA1, 0xC5, 0x99 };
static const symbol s_2_141[4] = { 0xC3, 0xA9, 0xC5, 0x99 };
static const symbol s_2_142[5] = { 'n', 0xC3, 0xA9, 0xC5, 0x99 };
static const symbol s_2_143[4] = { 0xC3, 0xAD, 0xC5, 0x99 };
static const symbol s_2_144[4] = { 0xC3, 0xBD, 0xC5, 0x99 };
static const symbol s_2_145[4] = { 'y', 'n', 0xC4, 0x9B };
static const symbol s_2_146[5] = { 'k', 'y', 'n', 0xC4, 0x9B };
static const symbol s_2_147[6] = { 'i', 0xC5, 0xA1, 't', 0xC4, 0x9B };
static const symbol s_2_148[8] = { 'o', 'v', 'i', 0xC5, 0xA1, 't', 0xC4, 0x9B };
static const symbol s_2_149[3] = { 'n', 0xC3, 0xA1 };
static const symbol s_2_150[4] = { 'o', 'v', 0xC3, 0xA1 };
static const symbol s_2_151[4] = { 'o', 'u', 0xC5, 0xA1 };
static const symbol s_2_152[4] = { 0xC3, 0xBD, 0xC5, 0xA1 };
static const symbol s_2_153[3] = { 'n', 0xC3, 0xA9 };
static const symbol s_2_154[5] = { 'o', 'v', 'n', 0xC3, 0xA9 };
static const symbol s_2_155[6] = { 0xC5, 0x99, 'a', 'd', 0xC3, 0xAD };
static const symbol s_2_156[3] = { 'n', 0xC3, 0xAD };
static const symbol s_2_157[4] = { 'a', 'n', 0xC3, 0xAD };
static const symbol s_2_158[4] = { 'e', 'n', 0xC3, 0xAD };
static const symbol s_2_159[5] = { 0xC4, 0x9B, 'n', 0xC3, 0xAD };
static const symbol s_2_160[5] = { 0xC3, 0xA1, 'n', 0xC3, 0xAD };
static const symbol s_2_161[3] = { 't', 0xC3, 0xAD };
static const symbol s_2_162[4] = { 'e', 't', 0xC3, 0xAD };
static const symbol s_2_163[5] = { 0xC4, 0x9B, 't', 0xC3, 0xAD };
static const symbol s_2_164[4] = { 'o', 'v', 0xC3, 0xAD };
static const symbol s_2_165[6] = { 's', 'l', 'o', 'v', 0xC3, 0xAD };
static const symbol s_2_166[5] = { 'c', 't', 'v', 0xC3, 0xAD };
static const symbol s_2_167[5] = { 's', 't', 'v', 0xC3, 0xAD };
static const symbol s_2_168[4] = { 0xC4, 0x8D, 0xC3, 0xAD };
static const symbol s_2_169[4] = { 0xC3, 0xA1, 0xC5, 0xBE };

static const struct among a_2[170] =
{
/*  0 */ { 2, s_2_0, -1, 3, 0},
/*  1 */ { 3, s_2_1, 0, 3, 0},
/*  2 */ { 3, s_2_2, 0, 3, 0},
/*  3 */ { 4, s_2_3, 2, 1, 0},
/*  4 */ { 3, s_2_4, -1, 3, 0},
/*  5 */ { 4, s_2_5, -1, 3, 0},
/*  6 */ { 2, s_2_6, -1, 3, 0},
/*  7 */ { 3, s_2_7, 6, 1, 0},
/*  8 */ { 3, s_2_8, 6, 3, 0},
/*  9 */ { 4, s_2_9, 8, 3, 0},
/* 10 */ { 4, s_2_10, 8, 1, 0},
/* 11 */ { 5, s_2_11, 8, 3, 0},
/* 12 */ { 3, s_2_12, 6, 3, 0},
/* 13 */ { 3, s_2_13, 6, 3, 0},
/* 14 */ { 4, s_2_14, 6, 3, 0},
/* 15 */ { 5, s_2_15, 14, 3, 0},
/* 16 */ { 6, s_2_16, 14, 3, 0},
/* 17 */ { 5, s_2_17, 6, 3, 0},
/* 18 */ { 5, s_2_18, 6, 3, 0},
/* 19 */ { 3, s_2_19, -1, 3, 0},
/* 20 */ { 4, s_2_20, -1, 3, 0},
/* 21 */ { 2, s_2_21, -1, 3, 0},
/* 22 */ { 3, s_2_22, 21, 1, 0},
/* 23 */ { 4, s_2_23, 22, 3, 0},
/* 24 */ { 6, s_2_24, 23, 3, 0},
/* 25 */ { 5, s_2_25, 22, 3, 0},
/* 26 */ { 5, s_2_26, 21, 3, 0},
/* 27 */ { 5, s_2_27, 21, 3, 0},
/* 28 */ { 5, s_2_28, 21, 3, 0},
/* 29 */ { 4, s_2_29, 21, 3, 0},
/* 30 */ { 4, s_2_30, 21, 1, 0},
/* 31 */ { 5, s_2_31, 21, 3, 0},
/* 32 */ { 3, s_2_32, -1, 3, 0},
/* 33 */ { 4, s_2_33, 32, 3, 0},
/* 34 */ { 3, s_2_34, -1, 1, 0},
/* 35 */ { 3, s_2_35, -1, 3, 0},
/* 36 */ { 4, s_2_36, -1, 1, 0},
/* 37 */ { 3, s_2_37, -1, 3, 0},
/* 38 */ { 2, s_2_38, -1, 3, 0},
/* 39 */ { 4, s_2_39, 38, 1, 0},
/* 40 */ { 6, s_2_40, 39, 3, 0},
/* 41 */ { 3, s_2_41, 38, 3, 0},
/* 42 */ { 4, s_2_42, 38, 3, 0},
/* 43 */ { 4, s_2_43, -1, 3, 0},
/* 44 */ { 3, s_2_44, -1, 3, 0},
/* 45 */ { 2, s_2_45, -1, 3, 0},
/* 46 */ { 3, s_2_46, 45, 3, 0},
/* 47 */ { 3, s_2_47, 45, 1, 0},
/* 48 */ { 4, s_2_48, 47, 3, 0},
/* 49 */ { 5, s_2_49, 48, 3, 0},
/* 50 */ { 3, s_2_50, 45, 3, 0},
/* 51 */ { 4, s_2_51, 50, 3, 0},
/* 52 */ { 4, s_2_52, 50, 3, 0},
/* 53 */ { 2, s_2_53, -1, 1, 0},
/* 54 */ { 4, s_2_54, 53, 3, 0},
/* 55 */ { 3, s_2_55, -1, 3, 0},
/* 56 */ { 2, s_2_56, -1, 3, 0},
/* 57 */ { 2, s_2_57, -1, 3, 0},
/* 58 */ { 2, s_2_58, -1, 3, 0},
/* 59 */ { 3, s_2_59, -1, 3, 0},
/* 60 */ { 4, s_2_60, -1, 3, 0},
/* 61 */ { 3, s_2_61, -1, 1, 0},
/* 62 */ { 3, s_2_62, -1, 3, 0},
/* 63 */ { 3, s_2_63, -1, 3, 0},
/* 64 */ { 2, s_2_64, -1, 3, 0},
/* 65 */ { 5, s_2_65, 64, 3, 0},
/* 66 */ { 5, s_2_66, 64, 3, 0},
/* 67 */ { 5, s_2_67, 64, 3, 0},
/* 68 */ { 5, s_2_68, 64, 3, 0},
/* 69 */ { 6, s_2_69, 64, 3, 0},
/* 70 */ { 6, s_2_70, 64, 3, 0},
/* 71 */ { 6, s_2_71, 64, 3, 0},
/* 72 */ { 2, s_2_72, -1, 1, 0},
/* 73 */ { 3, s_2_73, -1, 1, 0},
/* 74 */ { 3, s_2_74, -1, 3, 0},
/* 75 */ { 4, s_2_75, -1, 3, 0},
/* 76 */ { 3, s_2_76, -1, 3, 0},
/* 77 */ { 3, s_2_77, -1, 3, 0},
/* 78 */ { 4, s_2_78, 77, 2, 0},
/* 79 */ { 6, s_2_79, 78, 3, 0},
/* 80 */ { 3, s_2_80, -1, 3, 0},
/* 81 */ { 6, s_2_81, 80, 3, 0},
/* 82 */ { 4, s_2_82, 80, 1, 0},
/* 83 */ { 3, s_2_83, -1, 3, 0},
/* 84 */ { 3, s_2_84, -1, 3, 0},
/* 85 */ { 3, s_2_85, -1, 1, 0},
/* 86 */ { 4, s_2_86, -1, 1, 0},
/* 87 */ { 4, s_2_87, -1, 1, 0},
/* 88 */ { 2, s_2_88, -1, 3, 0},
/* 89 */ { 4, s_2_89, 88, 3, 0},
/* 90 */ { 3, s_2_90, -1, 3, 0},
/* 91 */ { 4, s_2_91, 90, 3, 0},
/* 92 */ { 3, s_2_92, -1, 3, 0},
/* 93 */ { 4, s_2_93, 92, 1, 0},
/* 94 */ { 3, s_2_94, -1, 3, 0},
/* 95 */ { 2, s_2_95, -1, 3, 0},
/* 96 */ { 4, s_2_96, 95, 1, 0},
/* 97 */ { 6, s_2_97, 96, 3, 0},
/* 98 */ { 3, s_2_98, 95, 3, 0},
/* 99 */ { 5, s_2_99, 98, 3, 0},
/*100 */ { 5, s_2_100, 95, 3, 0},
/*101 */ { 6, s_2_101, 95, 3, 0},
/*102 */ { 6, s_2_102, 95, 3, 0},
/*103 */ { 2, s_2_103, -1, 3, 0},
/*104 */ { 3, s_2_104, 103, 3, 0},
/*105 */ { 2, s_2_105, -1, 3, 0},
/*106 */ { 4, s_2_106, 105, 3, 0},
/*107 */ { 3, s_2_107, -1, 1, 0},
/*108 */ { 3, s_2_108, -1, 3, 0},
/*109 */ { 4, s_2_109, 108, 3, 0},
/*110 */ { 6, s_2_110, 109, 3, 0},
/*111 */ { 2, s_2_111, -1, 3, 0},
/*112 */ { 2, s_2_112, -1, 3, 0},
/*113 */ { 3, s_2_113, 112, 3, 0},
/*114 */ { 5, s_2_114, 113, 3, 0},
/*115 */ { 3, s_2_115, -1, 3, 0},
/*116 */ { 4, s_2_116, -1, 3, 0},
/*117 */ { 3, s_2_117, -1, 3, 0},
/*118 */ { 4, s_2_118, 117, 1, 0},
/*119 */ { 5, s_2_119, 117, 3, 0},
/*120 */ { 2, s_2_120, -1, 3, 0},
/*121 */ { 5, s_2_121, -1, 1, 0},
/*122 */ { 3, s_2_122, -1, 3, 0},
/*123 */ { 3, s_2_123, -1, 3, 0},
/*124 */ { 3, s_2_124, -1, 3, 0},
/*125 */ { 4, s_2_125, 124, 3, 0},
/*126 */ { 2, s_2_126, -1, 3, 0},
/*127 */ { 3, s_2_127, -1, 3, 0},
/*128 */ { 3, s_2_128, -1, 3, 0},
/*129 */ { 4, s_2_129, 128, 3, 0},
/*130 */ { 3, s_2_130, -1, 3, 0},
/*131 */ { 3, s_2_131, -1, 3, 0},
/*132 */ { 2, s_2_132, -1, 3, 0},
/*133 */ { 3, s_2_133, -1, 1, 0},
/*134 */ { 3, s_2_134, -1, 3, 0},
/*135 */ { 4, s_2_135, -1, 3, 0},
/*136 */ { 3, s_2_136, -1, 3, 0},
/*137 */ { 4, s_2_137, -1, 3, 0},
/*138 */ { 4, s_2_138, -1, 3, 0},
/*139 */ { 5, s_2_139, 138, 3, 0},
/*140 */ { 7, s_2_140, 138, 1, 0},
/*141 */ { 4, s_2_141, -1, 3, 0},
/*142 */ { 5, s_2_142, 141, 3, 0},
/*143 */ { 4, s_2_143, -1, 3, 0},
/*144 */ { 4, s_2_144, -1, 3, 0},
/*145 */ { 4, s_2_145, -1, 3, 0},
/*146 */ { 5, s_2_146, 145, 3, 0},
/*147 */ { 6, s_2_147, -1, 1, 0},
/*148 */ { 8, s_2_148, 147, 3, 0},
/*149 */ { 3, s_2_149, -1, 3, 0},
/*150 */ { 4, s_2_150, -1, 3, 0},
/*151 */ { 4, s_2_151, -1, 3, 0},
/*152 */ { 4, s_2_152, -1, 3, 0},
/*153 */ { 3, s_2_153, -1, 3, 0},
/*154 */ { 5, s_2_154, 153, 3, 0},
/*155 */ { 6, s_2_155, -1, 3, 0},
/*156 */ { 3, s_2_156, -1, 3, 0},
/*157 */ { 4, s_2_157, 156, 3, 0},
/*158 */ { 4, s_2_158, 156, 3, 0},
/*159 */ { 5, s_2_159, 156, 3, 0},
/*160 */ { 5, s_2_160, 156, 3, 0},
/*161 */ { 3, s_2_161, -1, 3, 0},
/*162 */ { 4, s_2_162, 161, 3, 0},
/*163 */ { 5, s_2_163, 161, 3, 0},
/*164 */ { 4, s_2_164, -1, 3, 0},
/*165 */ { 6, s_2_165, 164, 3, 0},
/*166 */ { 5, s_2_166, -1, 3, 0},
/*167 */ { 5, s_2_167, -1, 3, 0},
/*168 */ { 4, s_2_168, -1, 3, 0},
/*169 */ { 4, s_2_169, -1, 3, 0}
};

static const symbol s_3_0[1] = { 'a' };
static const symbol s_3_1[2] = { 'b', 'a' };
static const symbol s_3_2[2] = { 'g', 'a' };
static const symbol s_3_3[2] = { 'k', 'a' };
static const symbol s_3_4[2] = { 'l', 'a' };
static const symbol s_3_5[2] = { 'n', 'a' };
static const symbol s_3_6[2] = { 'r', 'a' };
static const symbol s_3_7[3] = { 'a', 't', 'a' };
static const symbol s_3_8[2] = { 'v', 'a' };
static const symbol s_3_9[2] = { 'e', 'b' };
static const symbol s_3_10[2] = { 'e', 'c' };
static const symbol s_3_11[1] = { 'e' };
static const symbol s_3_12[2] = { 'c', 'e' };
static const symbol s_3_13[2] = { 'l', 'e' };
static const symbol s_3_14[3] = { 'e', 't', 'e' };
static const symbol s_3_15[4] = { 0xC4, 0x9B, 't', 'e' };
static const symbol s_3_16[3] = { 0xC4, 0x8D, 'e' };
static const symbol s_3_17[1] = { 'g' };
static const symbol s_3_18[3] = { 'e', 'c', 'h' };
static const symbol s_3_19[4] = { 'l', 'e', 'c', 'h' };
static const symbol s_3_20[4] = { 'n', 'e', 'c', 'h' };
static const symbol s_3_21[5] = { 'a', 't', 'e', 'c', 'h' };
static const symbol s_3_22[4] = { 'v', 'e', 'c', 'h' };
static const symbol s_3_23[4] = { 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_24[5] = { 'b', 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_25[5] = { 'k', 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_26[5] = { 'n', 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_27[5] = { 'r', 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_28[5] = { 'v', 0xC3, 0xA1, 'c', 'h' };
static const symbol s_3_29[4] = { 0xC3, 0xAD, 'c', 'h' };
static const symbol s_3_30[5] = { 'c', 0xC3, 0xAD, 'c', 'h' };
static const symbol s_3_31[5] = { 'n', 0xC3, 0xAD, 'c', 'h' };
static const symbol s_3_32[5] = { 't', 0xC3, 0xAD, 'c', 'h' };
static const symbol s_3_33[5] = { 'z', 0xC3, 0xAD, 'c', 'h' };
static const symbol s_3_34[1] = { 'i' };
static const symbol s_3_35[2] = { 'c', 'i' };
static const symbol s_3_36[3] = { 'a', 'm', 'i' };
static const symbol s_3_37[4] = { 'b', 'a', 'm', 'i' };
static const symbol s_3_38[4] = { 'k', 'a', 'm', 'i' };
static const symbol s_3_39[4] = { 'n', 'a', 'm', 'i' };
static const symbol s_3_40[4] = { 'r', 'a', 'm', 'i' };
static const symbol s_3_41[4] = { 'v', 'a', 'm', 'i' };
static const symbol s_3_42[3] = { 'e', 'm', 'i' };
static const symbol s_3_43[4] = { 0xC4, 0x9B, 'm', 'i' };
static const symbol s_3_44[5] = { 'n', 0xC4, 0x9B, 'm', 'i' };
static const symbol s_3_45[2] = { 'n', 'i' };
static const symbol s_3_46[2] = { 't', 'i' };
static const symbol s_3_47[3] = { 'e', 't', 'i' };
static const symbol s_3_48[4] = { 0xC4, 0x9B, 't', 'i' };
static const symbol s_3_49[3] = { 'o', 'v', 'i' };
static const symbol s_3_50[4] = { 'g', 'o', 'v', 'i' };
static const symbol s_3_51[4] = { 'k', 'o', 'v', 'i' };
static const symbol s_3_52[2] = { 'z', 'i' };
static const symbol s_3_53[1] = { 'k' };
static const symbol s_3_54[2] = { 'e', 'k' };
static const symbol s_3_55[2] = { 'e', 'l' };
static const symbol s_3_56[2] = { 'e', 'm' };
static const symbol s_3_57[3] = { 'c', 'e', 'm' };
static const symbol s_3_58[3] = { 'g', 'e', 'm' };
static const symbol s_3_59[3] = { 'k', 'e', 'm' };
static const symbol s_3_60[3] = { 'l', 'e', 'm' };
static const symbol s_3_61[3] = { 'n', 'e', 'm' };
static const symbol s_3_62[4] = { 'e', 't', 'e', 'm' };
static const symbol s_3_63[5] = { 0xC4, 0x9B, 't', 'e', 'm' };
static const symbol s_3_64[3] = { 'v', 'e', 'm' };
static const symbol s_3_65[2] = { 'u', 'm' };
static const symbol s_3_66[3] = { 'k', 'u', 'm' };
static const symbol s_3_67[4] = { 't', 0xC4, 0x9B, 'm' };
static const symbol s_3_68[3] = { 0xC3, 0xA1, 'm' };
static const symbol s_3_69[4] = { 'b', 0xC3, 0xA1, 'm' };
static const symbol s_3_70[4] = { 'k', 0xC3, 0xA1, 'm' };
static const symbol s_3_71[4] = { 'n', 0xC3, 0xA1, 'm' };
static const symbol s_3_72[4] = { 'r', 0xC3, 0xA1, 'm' };
static const symbol s_3_73[4] = { 'v', 0xC3, 0xA1, 'm' };
static const symbol s_3_74[3] = { 0xC3, 0xAD, 'm' };
static const symbol s_3_75[4] = { 'n', 0xC3, 0xAD, 'm' };
static const symbol s_3_76[4] = { 't', 0xC3, 0xAD, 'm' };
static const symbol s_3_77[3] = { 0xC5, 0xAF, 'm' };
static const symbol s_3_78[4] = { 'c', 0xC5, 0xAF, 'm' };
static const symbol s_3_79[4] = { 'g', 0xC5, 0xAF, 'm' };
static const symbol s_3_80[4] = { 'k', 0xC5, 0xAF, 'm' };
static const symbol s_3_81[4] = { 'l', 0xC5, 0xAF, 'm' };
static const symbol s_3_82[4] = { 'n', 0xC5, 0xAF, 'm' };
static const symbol s_3_83[5] = { 'a', 't', 0xC5, 0xAF, 'm' };
static const symbol s_3_84[4] = { 'v', 0xC5, 0xAF, 'm' };
static const symbol s_3_85[2] = { 'e', 'n' };
static const symbol s_3_86[1] = { 'o' };
static const symbol s_3_87[2] = { 'b', 'o' };
static const symbol s_3_88[2] = { 'k', 'o' };
static const symbol s_3_89[2] = { 'l', 'o' };
static const symbol s_3_90[2] = { 'n', 'o' };
static const symbol s_3_91[2] = { 'r', 'o' };
static const symbol s_3_92[2] = { 'v', 'o' };
static const symbol s_3_93[1] = { 'r' };
static const symbol s_3_94[2] = { 'a', 't' };
static const symbol s_3_95[1] = { 'u' };
static const symbol s_3_96[2] = { 'b', 'u' };
static const symbol s_3_97[2] = { 'g', 'u' };
static const symbol s_3_98[2] = { 'k', 'u' };
static const symbol s_3_99[2] = { 'l', 'u' };
static const symbol s_3_100[2] = { 'n', 'u' };
static const symbol s_3_101[2] = { 'o', 'u' };
static const symbol s_3_102[3] = { 'b', 'o', 'u' };
static const symbol s_3_103[3] = { 'k', 'o', 'u' };
static const symbol s_3_104[3] = { 'n', 'o', 'u' };
static const symbol s_3_105[3] = { 'r', 'o', 'u' };
static const symbol s_3_106[3] = { 'v', 'o', 'u' };
static const symbol s_3_107[2] = { 'r', 'u' };
static const symbol s_3_108[2] = { 'v', 'u' };
static const symbol s_3_109[2] = { 'e', 'v' };
static const symbol s_3_110[1] = { 'y' };
static const symbol s_3_111[2] = { 'b', 'y' };
static const symbol s_3_112[2] = { 'g', 'y' };
static const symbol s_3_113[2] = { 'k', 'y' };
static const symbol s_3_114[2] = { 'l', 'y' };
static const symbol s_3_115[2] = { 'n', 'y' };
static const symbol s_3_116[2] = { 'r', 'y' };
static const symbol s_3_117[3] = { 'a', 't', 'y' };
static const symbol s_3_118[2] = { 'v', 'y' };
static const symbol s_3_119[2] = { 0xC5, 0x88 };
static const symbol s_3_120[3] = { 'e', 0xC5, 0x88 };
static const symbol s_3_121[2] = { 0xC4, 0x9B };
static const symbol s_3_122[3] = { 'b', 0xC4, 0x9B };
static const symbol s_3_123[3] = { 'n', 0xC4, 0x9B };
static const symbol s_3_124[3] = { 't', 0xC4, 0x9B };
static const symbol s_3_125[3] = { 'v', 0xC4, 0x9B };
static const symbol s_3_126[2] = { 0xC5, 0xA5 };
static const symbol s_3_127[2] = { 0xC3, 0xA9 };
static const symbol s_3_128[4] = { 'o', 'v', 0xC3, 0xA9 };
static const symbol s_3_129[5] = { 'g', 'o', 'v', 0xC3, 0xA9 };
static const symbol s_3_130[2] = { 0xC3, 0xAD };
static const symbol s_3_131[3] = { 'n', 0xC3, 0xAD };
static const symbol s_3_132[2] = { 0xC5, 0xAF };
static const symbol s_3_133[3] = { 'c', 0xC5, 0xAF };
static const symbol s_3_134[3] = { 'g', 0xC5, 0xAF };
static const symbol s_3_135[3] = { 'k', 0xC5, 0xAF };

static const struct among a_3[136] =
{
/*  0 */ { 1, s_3_0, -1, 17, 0},
/*  1 */ { 2, s_3_1, 0, 17, 0},
/*  2 */ { 2, s_3_2, 0, 11, 0},
/*  3 */ { 2, s_3_3, 0, 17, 0},
/*  4 */ { 2, s_3_4, 0, 3, 0},
/*  5 */ { 2, s_3_5, 0, 17, 0},
/*  6 */ { 2, s_3_6, 0, 17, 0},
/*  7 */ { 3, s_3_7, 0, 17, 0},
/*  8 */ { 2, s_3_8, 0, 17, 0},
/*  9 */ { 2, s_3_9, -1, 1, 0},
/* 10 */ { 2, s_3_10, -1, 17, 0},
/* 11 */ { 1, s_3_11, -1, 17, 0},
/* 12 */ { 2, s_3_12, 11, 17, 0},
/* 13 */ { 2, s_3_13, 11, 3, 0},
/* 14 */ { 3, s_3_14, 11, 6, 0},
/* 15 */ { 4, s_3_15, 11, 15, 0},
/* 16 */ { 3, s_3_16, 11, 2, 0},
/* 17 */ { 1, s_3_17, -1, 17, 0},
/* 18 */ { 3, s_3_18, -1, 17, 0},
/* 19 */ { 4, s_3_19, 18, 3, 0},
/* 20 */ { 4, s_3_20, 18, 7, 0},
/* 21 */ { 5, s_3_21, 18, 17, 0},
/* 22 */ { 4, s_3_22, 18, 12, 0},
/* 23 */ { 4, s_3_23, -1, 13, 0},
/* 24 */ { 5, s_3_24, 23, 1, 0},
/* 25 */ { 5, s_3_25, 23, 17, 0},
/* 26 */ { 5, s_3_26, 23, 5, 0},
/* 27 */ { 5, s_3_27, 23, 4, 0},
/* 28 */ { 5, s_3_28, 23, 8, 0},
/* 29 */ { 4, s_3_29, -1, 17, 0},
/* 30 */ { 5, s_3_30, 29, 17, 0},
/* 31 */ { 5, s_3_31, 29, 17, 0},
/* 32 */ { 5, s_3_32, 29, 10, 0},
/* 33 */ { 5, s_3_33, 29, 17, 0},
/* 34 */ { 1, s_3_34, -1, 17, 0},
/* 35 */ { 2, s_3_35, 34, 17, 0},
/* 36 */ { 3, s_3_36, 34, 13, 0},
/* 37 */ { 4, s_3_37, 36, 1, 0},
/* 38 */ { 4, s_3_38, 36, 16, 0},
/* 39 */ { 4, s_3_39, 36, 5, 0},
/* 40 */ { 4, s_3_40, 36, 4, 0},
/* 41 */ { 4, s_3_41, 36, 8, 0},
/* 42 */ { 3, s_3_42, 34, 17, 0},
/* 43 */ { 4, s_3_43, 34, 15, 0},
/* 44 */ { 5, s_3_44, 43, 14, 0},
/* 45 */ { 2, s_3_45, 34, 17, 0},
/* 46 */ { 2, s_3_46, 34, 10, 0},
/* 47 */ { 3, s_3_47, 46, 6, 0},
/* 48 */ { 4, s_3_48, 46, 15, 0},
/* 49 */ { 3, s_3_49, 34, 17, 0},
/* 50 */ { 4, s_3_50, 49, 11, 0},
/* 51 */ { 4, s_3_51, 49, 9, 0},
/* 52 */ { 2, s_3_52, 34, 17, 0},
/* 53 */ { 1, s_3_53, -1, 17, 0},
/* 54 */ { 2, s_3_54, 53, 17, 0},
/* 55 */ { 2, s_3_55, -1, 3, 0},
/* 56 */ { 2, s_3_56, -1, 17, 0},
/* 57 */ { 3, s_3_57, 56, 2, 0},
/* 58 */ { 3, s_3_58, 56, 11, 0},
/* 59 */ { 3, s_3_59, 56, 17, 0},
/* 60 */ { 3, s_3_60, 56, 3, 0},
/* 61 */ { 3, s_3_61, 56, 7, 0},
/* 62 */ { 4, s_3_62, 56, 6, 0},
/* 63 */ { 5, s_3_63, 56, 15, 0},
/* 64 */ { 3, s_3_64, 56, 12, 0},
/* 65 */ { 2, s_3_65, -1, 17, 0},
/* 66 */ { 3, s_3_66, 65, 17, 0},
/* 67 */ { 4, s_3_67, -1, 10, 0},
/* 68 */ { 3, s_3_68, -1, 13, 0},
/* 69 */ { 4, s_3_69, 68, 1, 0},
/* 70 */ { 4, s_3_70, 68, 16, 0},
/* 71 */ { 4, s_3_71, 68, 5, 0},
/* 72 */ { 4, s_3_72, 68, 4, 0},
/* 73 */ { 4, s_3_73, 68, 8, 0},
/* 74 */ { 3, s_3_74, -1, 17, 0},
/* 75 */ { 4, s_3_75, 74, 17, 0},
/* 76 */ { 4, s_3_76, 74, 10, 0},
/* 77 */ { 3, s_3_77, -1, 17, 0},
/* 78 */ { 4, s_3_78, 77, 2, 0},
/* 79 */ { 4, s_3_79, 77, 11, 0},
/* 80 */ { 4, s_3_80, 77, 17, 0},
/* 81 */ { 4, s_3_81, 77, 3, 0},
/* 82 */ { 4, s_3_82, 77, 7, 0},
/* 83 */ { 5, s_3_83, 77, 17, 0},
/* 84 */ { 4, s_3_84, 77, 12, 0},
/* 85 */ { 2, s_3_85, -1, 17, 0},
/* 86 */ { 1, s_3_86, -1, 17, 0},
/* 87 */ { 2, s_3_87, 86, 1, 0},
/* 88 */ { 2, s_3_88, 86, 17, 0},
/* 89 */ { 2, s_3_89, 86, 17, 0},
/* 90 */ { 2, s_3_90, 86, 17, 0},
/* 91 */ { 2, s_3_91, 86, 4, 0},
/* 92 */ { 2, s_3_92, 86, 17, 0},
/* 93 */ { 1, s_3_93, -1, 4, 0},
/* 94 */ { 2, s_3_94, -1, 17, 0},
/* 95 */ { 1, s_3_95, -1, 17, 0},
/* 96 */ { 2, s_3_96, 95, 1, 0},
/* 97 */ { 2, s_3_97, 95, 11, 0},
/* 98 */ { 2, s_3_98, 95, 17, 0},
/* 99 */ { 2, s_3_99, 95, 3, 0},
/*100 */ { 2, s_3_100, 95, 17, 0},
/*101 */ { 2, s_3_101, 95, 13, 0},
/*102 */ { 3, s_3_102, 101, 1, 0},
/*103 */ { 3, s_3_103, 101, 16, 0},
/*104 */ { 3, s_3_104, 101, 5, 0},
/*105 */ { 3, s_3_105, 101, 4, 0},
/*106 */ { 3, s_3_106, 101, 8, 0},
/*107 */ { 2, s_3_107, 95, 4, 0},
/*108 */ { 2, s_3_108, 95, 17, 0},
/*109 */ { 2, s_3_109, -1, 17, 0},
/*110 */ { 1, s_3_110, -1, 17, 0},
/*111 */ { 2, s_3_111, 110, 1, 0},
/*112 */ { 2, s_3_112, 110, 11, 0},
/*113 */ { 2, s_3_113, 110, 17, 0},
/*114 */ { 2, s_3_114, 110, 3, 0},
/*115 */ { 2, s_3_115, 110, 17, 0},
/*116 */ { 2, s_3_116, 110, 4, 0},
/*117 */ { 3, s_3_117, 110, 17, 0},
/*118 */ { 2, s_3_118, 110, 17, 0},
/*119 */ { 2, s_3_119, -1, 17, 0},
/*120 */ { 3, s_3_120, 119, 17, 0},
/*121 */ { 2, s_3_121, -1, 17, 0},
/*122 */ { 3, s_3_122, 121, 1, 0},
/*123 */ { 3, s_3_123, 121, 17, 0},
/*124 */ { 3, s_3_124, 121, 17, 0},
/*125 */ { 3, s_3_125, 121, 8, 0},
/*126 */ { 2, s_3_126, -1, 10, 0},
/*127 */ { 2, s_3_127, -1, 17, 0},
/*128 */ { 4, s_3_128, 127, 17, 0},
/*129 */ { 5, s_3_129, 128, 11, 0},
/*130 */ { 2, s_3_130, -1, 17, 0},
/*131 */ { 3, s_3_131, 130, 17, 0},
/*132 */ { 2, s_3_132, -1, 17, 0},
/*133 */ { 3, s_3_133, 132, 2, 0},
/*134 */ { 3, s_3_134, 132, 11, 0},
/*135 */ { 3, s_3_135, 132, 17, 0}
};

static const symbol s_4_0[3] = { 'i', 'n', 'a' };
static const symbol s_4_1[3] = { 'o', 'v', 'a' };
static const symbol s_4_2[4] = { 'o', 'v', 'i', 'c' };
static const symbol s_4_3[2] = { 'i', 'n' };
static const symbol s_4_4[3] = { 'i', 'n', 'o' };
static const symbol s_4_5[3] = { 'o', 'v', 'o' };
static const symbol s_4_6[3] = { 0xC5, 0xAF, 'v' };
static const symbol s_4_7[3] = { 'c', 0xC3, 0xAD };
static const symbol s_4_8[4] = { 'a', 'c', 0xC3, 0xAD };
static const symbol s_4_9[4] = { 'i', 'c', 0xC3, 0xAD };
static const symbol s_4_10[3] = { 'n', 0xC3, 0xAD };
static const symbol s_4_11[6] = { 'b', 'i', 'l', 'n', 0xC3, 0xAD };
static const symbol s_4_12[6] = { 0xC3, 0xA1, 'l', 'n', 0xC3, 0xAD };
static const symbol s_4_13[6] = { 0xC3, 0xA1, 'r', 'n', 0xC3, 0xAD };
static const symbol s_4_14[4] = { 't', 'n', 0xC3, 0xAD };
static const symbol s_4_15[5] = { 'n', 't', 'n', 0xC3, 0xAD };
static const symbol s_4_16[5] = { 'i', 'v', 'n', 0xC3, 0xAD };
static const symbol s_4_17[5] = { 'o', 'v', 'n', 0xC3, 0xAD };
static const symbol s_4_18[6] = { 'a', 0xC4, 0x8D, 'n', 0xC3, 0xAD };
static const symbol s_4_19[6] = { 'e', 0xC4, 0x8D, 'n', 0xC3, 0xAD };
static const symbol s_4_20[6] = { 'i', 0xC4, 0x8D, 'n', 0xC3, 0xAD };
static const symbol s_4_21[4] = { 'o', 'v', 0xC3, 0xAD };
static const symbol s_4_22[4] = { 0xC4, 0x8D, 0xC3, 0xAD };
static const symbol s_4_23[4] = { 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_24[6] = { 'e', 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_25[7] = { 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_26[8] = { 'n', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_27[10] = { 'o', 'v', 'n', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_28[8] = { 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_29[10] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_30[12] = { 'i', 'j', 0xC5, 0xA1, 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_31[12] = { 'o', 'v', 0xC5, 0xA1, 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_32[9] = { 'o', 'v', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_4_33[3] = { 'k', 0xC3, 0xBD };
static const symbol s_4_34[4] = { 'c', 'k', 0xC3, 0xBD };
static const symbol s_4_35[5] = { 'i', 'c', 'k', 0xC3, 0xBD };
static const symbol s_4_36[5] = { 'i', 'n', 'k', 0xC3, 0xBD };
static const symbol s_4_37[7] = { 'i', 'l', 'i', 'n', 'k', 0xC3, 0xBD };
static const symbol s_4_38[8] = { 'o', 'u', 'l', 'i', 'n', 'k', 0xC3, 0xBD };
static const symbol s_4_39[8] = { 'o', 'u', 'n', 'i', 'n', 'k', 0xC3, 0xBD };
static const symbol s_4_40[6] = { 'o', 'u', 'n', 'k', 0xC3, 0xBD };
static const symbol s_4_41[4] = { 's', 'k', 0xC3, 0xBD };
static const symbol s_4_42[6] = { 'i', 'j', 's', 'k', 0xC3, 0xBD };
static const symbol s_4_43[7] = { 0xC3, 0xA1, 'n', 's', 'k', 0xC3, 0xBD };
static const symbol s_4_44[6] = { 'o', 'v', 's', 'k', 0xC3, 0xBD };
static const symbol s_4_45[6] = { 'i', 0xC4, 0x8D, 'k', 0xC3, 0xBD };
static const symbol s_4_46[9] = { 'i', 0xC4, 0x8D, 'i', 0xC4, 0x8D, 'k', 0xC3, 0xBD };
static const symbol s_4_47[7] = { 'o', 'u', 0xC4, 0x8D, 'k', 0xC3, 0xBD };
static const symbol s_4_48[3] = { 'l', 0xC3, 0xBD };
static const symbol s_4_49[7] = { 'i', 0xC4, 0x8D, 'e', 'l', 0xC3, 0xBD };
static const symbol s_4_50[4] = { 'i', 'l', 0xC3, 0xBD };
static const symbol s_4_51[3] = { 'n', 0xC3, 0xBD };
static const symbol s_4_52[6] = { 'o', 'v', 'a', 'n', 0xC3, 0xBD };
static const symbol s_4_53[4] = { 'c', 'n', 0xC3, 0xBD };
static const symbol s_4_54[4] = { 'e', 'n', 0xC3, 0xBD };
static const symbol s_4_55[6] = { 't', 'e', 'l', 'n', 0xC3, 0xBD };
static const symbol s_4_56[7] = { 'i', 't', 'e', 'l', 'n', 0xC3, 0xBD };
static const symbol s_4_57[4] = { 't', 'n', 0xC3, 0xBD };
static const symbol s_4_58[5] = { 'u', 't', 'n', 0xC3, 0xBD };
static const symbol s_4_59[6] = { 'e', 0xC4, 0x8D, 'n', 0xC3, 0xBD };
static const symbol s_4_60[6] = { 'i', 0xC4, 0x8D, 'n', 0xC3, 0xBD };
static const symbol s_4_61[5] = { 0xC4, 0x9B, 'n', 0xC3, 0xBD };
static const symbol s_4_62[4] = { 'a', 't', 0xC3, 0xBD };
static const symbol s_4_63[7] = { 'i', 'n', 'k', 'a', 't', 0xC3, 0xBD };
static const symbol s_4_64[5] = { 'n', 'a', 't', 0xC3, 0xBD };
static const symbol s_4_65[4] = { 'i', 't', 0xC3, 0xBD };
static const symbol s_4_66[6] = { 'o', 'v', 'i', 't', 0xC3, 0xBD };
static const symbol s_4_67[7] = { 'i', 0xC4, 0x8D, 'i', 't', 0xC3, 0xBD };
static const symbol s_4_68[5] = { 'i', 's', 't', 0xC3, 0xBD };
static const symbol s_4_69[4] = { 'u', 't', 0xC3, 0xBD };
static const symbol s_4_70[3] = { 'v', 0xC3, 0xBD };
static const symbol s_4_71[4] = { 'a', 'v', 0xC3, 0xBD };
static const symbol s_4_72[5] = { 'k', 'a', 'v', 0xC3, 0xBD };
static const symbol s_4_73[5] = { 'l', 'a', 'v', 0xC3, 0xBD };
static const symbol s_4_74[4] = { 'i', 'v', 0xC3, 0xBD };
static const symbol s_4_75[5] = { 'l', 'i', 'v', 0xC3, 0xBD };
static const symbol s_4_76[4] = { 'o', 'v', 0xC3, 0xBD };

static const struct among a_4[77] =
{
/*  0 */ { 3, s_4_0, -1, 2, 0},
/*  1 */ { 3, s_4_1, -1, 1, 0},
/*  2 */ { 4, s_4_2, -1, 1, 0},
/*  3 */ { 2, s_4_3, -1, 2, 0},
/*  4 */ { 3, s_4_4, -1, 2, 0},
/*  5 */ { 3, s_4_5, -1, 1, 0},
/*  6 */ { 3, s_4_6, -1, 1, 0},
/*  7 */ { 3, s_4_7, -1, 1, 0},
/*  8 */ { 4, s_4_8, 7, 1, 0},
/*  9 */ { 4, s_4_9, 7, 1, 0},
/* 10 */ { 3, s_4_10, -1, 4, 0},
/* 11 */ { 6, s_4_11, 10, 1, 0},
/* 12 */ { 6, s_4_12, 10, 1, 0},
/* 13 */ { 6, s_4_13, 10, 1, 0},
/* 14 */ { 4, s_4_14, 10, 1, 0},
/* 15 */ { 5, s_4_15, 14, 1, 0},
/* 16 */ { 5, s_4_16, 10, 1, 0},
/* 17 */ { 5, s_4_17, 10, 1, 0},
/* 18 */ { 6, s_4_18, 10, 1, 0},
/* 19 */ { 6, s_4_19, 10, 1, 0},
/* 20 */ { 6, s_4_20, 10, 1, 0},
/* 21 */ { 4, s_4_21, -1, 1, 0},
/* 22 */ { 4, s_4_22, -1, 1, 0},
/* 23 */ { 4, s_4_23, -1, 1, 0},
/* 24 */ { 6, s_4_24, 23, 6, 0},
/* 25 */ { 7, s_4_25, 23, 6, 0},
/* 26 */ { 8, s_4_26, 25, 1, 0},
/* 27 */ { 10, s_4_27, 26, 1, 0},
/* 28 */ { 8, s_4_28, 25, 1, 0},
/* 29 */ { 10, s_4_29, 28, 3, 0},
/* 30 */ { 12, s_4_30, 29, 3, 0},
/* 31 */ { 12, s_4_31, 29, 1, 0},
/* 32 */ { 9, s_4_32, 25, 1, 0},
/* 33 */ { 3, s_4_33, -1, 3, 0},
/* 34 */ { 4, s_4_34, 33, 1, 0},
/* 35 */ { 5, s_4_35, 34, 1, 0},
/* 36 */ { 5, s_4_36, 33, 2, 0},
/* 37 */ { 7, s_4_37, 36, 1, 0},
/* 38 */ { 8, s_4_38, 36, 1, 0},
/* 39 */ { 8, s_4_39, 36, 1, 0},
/* 40 */ { 6, s_4_40, 33, 1, 0},
/* 41 */ { 4, s_4_41, 33, 3, 0},
/* 42 */ { 6, s_4_42, 41, 1, 0},
/* 43 */ { 7, s_4_43, 41, 1, 0},
/* 44 */ { 6, s_4_44, 41, 3, 0},
/* 45 */ { 6, s_4_45, 33, 2, 0},
/* 46 */ { 9, s_4_46, 45, 1, 0},
/* 47 */ { 7, s_4_47, 33, 1, 0},
/* 48 */ { 3, s_4_48, -1, 1, 0},
/* 49 */ { 7, s_4_49, 48, 1, 0},
/* 50 */ { 4, s_4_50, 48, 1, 0},
/* 51 */ { 3, s_4_51, -1, 4, 0},
/* 52 */ { 6, s_4_52, 51, 1, 0},
/* 53 */ { 4, s_4_53, 51, 1, 0},
/* 54 */ { 4, s_4_54, 51, 1, 0},
/* 55 */ { 6, s_4_55, 51, 1, 0},
/* 56 */ { 7, s_4_56, 55, 4, 0},
/* 57 */ { 4, s_4_57, 51, 1, 0},
/* 58 */ { 5, s_4_58, 57, 1, 0},
/* 59 */ { 6, s_4_59, 51, 1, 0},
/* 60 */ { 6, s_4_60, 51, 1, 0},
/* 61 */ { 5, s_4_61, 51, 1, 0},
/* 62 */ { 4, s_4_62, -1, 1, 0},
/* 63 */ { 7, s_4_63, 62, 1, 0},
/* 64 */ { 5, s_4_64, 62, 5, 0},
/* 65 */ { 4, s_4_65, -1, 2, 0},
/* 66 */ { 6, s_4_66, 65, 1, 0},
/* 67 */ { 7, s_4_67, 65, 1, 0},
/* 68 */ { 5, s_4_68, -1, 1, 0},
/* 69 */ { 4, s_4_69, -1, 1, 0},
/* 70 */ { 3, s_4_70, -1, 1, 0},
/* 71 */ { 4, s_4_71, 70, 1, 0},
/* 72 */ { 5, s_4_72, 71, 1, 0},
/* 73 */ { 5, s_4_73, 71, 1, 0},
/* 74 */ { 4, s_4_74, 70, 2, 0},
/* 75 */ { 5, s_4_75, 74, 1, 0},
/* 76 */ { 4, s_4_76, 70, 1, 0}
};

static const symbol s_5_0[3] = { 'i', 'n', 'a' };
static const symbol s_5_1[3] = { 'o', 'v', 'a' };
static const symbol s_5_2[4] = { 0xC3, 0xAD, 'c', 'h' };
static const symbol s_5_3[4] = { 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_4[5] = { 'h', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_5[6] = { 'c', 'h', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_6[5] = { 'k', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_7[6] = { 'c', 'k', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_8[6] = { 's', 'k', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_9[6] = { 'i', 'n', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_10[5] = { 'r', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_11[6] = { 'o', 'v', 0xC3, 0xBD, 'c', 'h' };
static const symbol s_5_12[4] = { 0xC3, 0xAD, 'm', 'i' };
static const symbol s_5_13[4] = { 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_14[5] = { 'h', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_15[6] = { 'c', 'h', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_16[5] = { 'k', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_17[6] = { 'c', 'k', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_18[6] = { 's', 'k', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_19[6] = { 'i', 'n', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_20[5] = { 'r', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_21[6] = { 'o', 'v', 0xC3, 0xBD, 'm', 'i' };
static const symbol s_5_22[3] = { 'i', 'n', 'i' };
static const symbol s_5_23[3] = { 'o', 'v', 'i' };
static const symbol s_5_24[3] = { 0xC3, 0xA9, 'm' };
static const symbol s_5_25[4] = { 'h', 0xC3, 0xA9, 'm' };
static const symbol s_5_26[5] = { 'c', 'h', 0xC3, 0xA9, 'm' };
static const symbol s_5_27[4] = { 'k', 0xC3, 0xA9, 'm' };
static const symbol s_5_28[5] = { 'c', 'k', 0xC3, 0xA9, 'm' };
static const symbol s_5_29[5] = { 's', 'k', 0xC3, 0xA9, 'm' };
static const symbol s_5_30[4] = { 'r', 0xC3, 0xA9, 'm' };
static const symbol s_5_31[3] = { 0xC3, 0xAD, 'm' };
static const symbol s_5_32[3] = { 0xC3, 0xBD, 'm' };
static const symbol s_5_33[4] = { 'h', 0xC3, 0xBD, 'm' };
static const symbol s_5_34[5] = { 'c', 'h', 0xC3, 0xBD, 'm' };
static const symbol s_5_35[4] = { 'k', 0xC3, 0xBD, 'm' };
static const symbol s_5_36[5] = { 'c', 'k', 0xC3, 0xBD, 'm' };
static const symbol s_5_37[5] = { 's', 'k', 0xC3, 0xBD, 'm' };
static const symbol s_5_38[5] = { 'i', 'n', 0xC3, 0xBD, 'm' };
static const symbol s_5_39[4] = { 'r', 0xC3, 0xBD, 'm' };
static const symbol s_5_40[5] = { 'o', 'v', 0xC3, 0xBD, 'm' };
static const symbol s_5_41[2] = { 'i', 'n' };
static const symbol s_5_42[4] = { 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_43[5] = { 'h', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_44[6] = { 'c', 'h', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_45[5] = { 'k', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_46[6] = { 'c', 'k', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_47[6] = { 's', 'k', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_48[5] = { 'r', 0xC3, 0xA9, 'h', 'o' };
static const symbol s_5_49[4] = { 0xC3, 0xAD, 'h', 'o' };
static const symbol s_5_50[3] = { 'i', 'n', 'o' };
static const symbol s_5_51[3] = { 'o', 'v', 'o' };
static const symbol s_5_52[4] = { 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_53[5] = { 'h', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_54[6] = { 'c', 'h', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_55[5] = { 'k', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_56[6] = { 'c', 'k', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_57[6] = { 's', 'k', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_58[5] = { 'r', 0xC3, 0xA9, 'm', 'u' };
static const symbol s_5_59[4] = { 0xC3, 0xAD, 'm', 'u' };
static const symbol s_5_60[3] = { 'i', 'n', 'u' };
static const symbol s_5_61[2] = { 'o', 'u' };
static const symbol s_5_62[3] = { 'h', 'o', 'u' };
static const symbol s_5_63[4] = { 'c', 'h', 'o', 'u' };
static const symbol s_5_64[3] = { 'k', 'o', 'u' };
static const symbol s_5_65[4] = { 'c', 'k', 'o', 'u' };
static const symbol s_5_66[4] = { 's', 'k', 'o', 'u' };
static const symbol s_5_67[4] = { 'i', 'n', 'o', 'u' };
static const symbol s_5_68[3] = { 'r', 'o', 'u' };
static const symbol s_5_69[4] = { 'o', 'v', 'o', 'u' };
static const symbol s_5_70[3] = { 'o', 'v', 'u' };
static const symbol s_5_71[3] = { 0xC5, 0xAF, 'v' };
static const symbol s_5_72[3] = { 'i', 'n', 'y' };
static const symbol s_5_73[3] = { 'o', 'v', 'y' };
static const symbol s_5_74[4] = { 'i', 'n', 0xC4, 0x9B };
static const symbol s_5_75[4] = { 'o', 'v', 0xC4, 0x9B };
static const symbol s_5_76[2] = { 0xC3, 0xA1 };
static const symbol s_5_77[3] = { 'h', 0xC3, 0xA1 };
static const symbol s_5_78[4] = { 'c', 'h', 0xC3, 0xA1 };
static const symbol s_5_79[3] = { 'k', 0xC3, 0xA1 };
static const symbol s_5_80[4] = { 'c', 'k', 0xC3, 0xA1 };
static const symbol s_5_81[4] = { 's', 'k', 0xC3, 0xA1 };
static const symbol s_5_82[3] = { 'r', 0xC3, 0xA1 };
static const symbol s_5_83[2] = { 0xC3, 0xA9 };
static const symbol s_5_84[3] = { 'h', 0xC3, 0xA9 };
static const symbol s_5_85[4] = { 'c', 'h', 0xC3, 0xA9 };
static const symbol s_5_86[3] = { 'k', 0xC3, 0xA9 };
static const symbol s_5_87[4] = { 'c', 'k', 0xC3, 0xA9 };
static const symbol s_5_88[4] = { 's', 'k', 0xC3, 0xA9 };
static const symbol s_5_89[3] = { 'r', 0xC3, 0xA9 };
static const symbol s_5_90[2] = { 0xC3, 0xAD };
static const symbol s_5_91[3] = { 'c', 0xC3, 0xAD };
static const symbol s_5_92[5] = { 0xC4, 0x8D, 't', 0xC3, 0xAD };
static const symbol s_5_93[5] = { 0xC5, 0xA1, 't', 0xC3, 0xAD };
static const symbol s_5_94[4] = { 'a', 'v', 0xC3, 0xAD };
static const symbol s_5_95[4] = { 0xC5, 0x99, 0xC3, 0xAD };
static const symbol s_5_96[6] = { 'e', 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_97[8] = { 0xC5, 0x99, 'e', 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_98[7] = { 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_99[10] = { 0xC4, 0x8D, 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_100[10] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'j', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_101[5] = { 'r', 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_102[6] = { 0xC5, 0xA1, 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_103[6] = { 0xC5, 0xBE, 0xC5, 0xA1, 0xC3, 0xAD };
static const symbol s_5_104[2] = { 0xC3, 0xBD };
static const symbol s_5_105[3] = { 'h', 0xC3, 0xBD };
static const symbol s_5_106[4] = { 'c', 'h', 0xC3, 0xBD };
static const symbol s_5_107[3] = { 'k', 0xC3, 0xBD };
static const symbol s_5_108[4] = { 'c', 'k', 0xC3, 0xBD };
static const symbol s_5_109[4] = { 's', 'k', 0xC3, 0xBD };
static const symbol s_5_110[3] = { 'r', 0xC3, 0xBD };

static const struct among a_5[111] =
{
/*  0 */ { 3, s_5_0, -1, 11, 0},
/*  1 */ { 3, s_5_1, -1, 9, 0},
/*  2 */ { 4, s_5_2, -1, 8, 0},
/*  3 */ { 4, s_5_3, -1, 10, 0},
/*  4 */ { 5, s_5_4, 3, 5, 0},
/*  5 */ { 6, s_5_5, 4, 4, 0},
/*  6 */ { 5, s_5_6, 3, 3, 0},
/*  7 */ { 6, s_5_7, 6, 2, 0},
/*  8 */ { 6, s_5_8, 6, 1, 0},
/*  9 */ { 6, s_5_9, 3, 11, 0},
/* 10 */ { 5, s_5_10, 3, 6, 0},
/* 11 */ { 6, s_5_11, 3, 9, 0},
/* 12 */ { 4, s_5_12, -1, 8, 0},
/* 13 */ { 4, s_5_13, -1, 10, 0},
/* 14 */ { 5, s_5_14, 13, 5, 0},
/* 15 */ { 6, s_5_15, 14, 4, 0},
/* 16 */ { 5, s_5_16, 13, 3, 0},
/* 17 */ { 6, s_5_17, 16, 2, 0},
/* 18 */ { 6, s_5_18, 16, 1, 0},
/* 19 */ { 6, s_5_19, 13, 11, 0},
/* 20 */ { 5, s_5_20, 13, 6, 0},
/* 21 */ { 6, s_5_21, 13, 9, 0},
/* 22 */ { 3, s_5_22, -1, 11, 0},
/* 23 */ { 3, s_5_23, -1, 9, 0},
/* 24 */ { 3, s_5_24, -1, 10, 0},
/* 25 */ { 4, s_5_25, 24, 5, 0},
/* 26 */ { 5, s_5_26, 25, 4, 0},
/* 27 */ { 4, s_5_27, 24, 3, 0},
/* 28 */ { 5, s_5_28, 27, 2, 0},
/* 29 */ { 5, s_5_29, 27, 1, 0},
/* 30 */ { 4, s_5_30, 24, 6, 0},
/* 31 */ { 3, s_5_31, -1, 8, 0},
/* 32 */ { 3, s_5_32, -1, 10, 0},
/* 33 */ { 4, s_5_33, 32, 5, 0},
/* 34 */ { 5, s_5_34, 33, 4, 0},
/* 35 */ { 4, s_5_35, 32, 3, 0},
/* 36 */ { 5, s_5_36, 35, 2, 0},
/* 37 */ { 5, s_5_37, 35, 1, 0},
/* 38 */ { 5, s_5_38, 32, 11, 0},
/* 39 */ { 4, s_5_39, 32, 6, 0},
/* 40 */ { 5, s_5_40, 32, 9, 0},
/* 41 */ { 2, s_5_41, -1, 11, 0},
/* 42 */ { 4, s_5_42, -1, 10, 0},
/* 43 */ { 5, s_5_43, 42, 5, 0},
/* 44 */ { 6, s_5_44, 43, 4, 0},
/* 45 */ { 5, s_5_45, 42, 3, 0},
/* 46 */ { 6, s_5_46, 45, 2, 0},
/* 47 */ { 6, s_5_47, 45, 1, 0},
/* 48 */ { 5, s_5_48, 42, 6, 0},
/* 49 */ { 4, s_5_49, -1, 8, 0},
/* 50 */ { 3, s_5_50, -1, 11, 0},
/* 51 */ { 3, s_5_51, -1, 9, 0},
/* 52 */ { 4, s_5_52, -1, 10, 0},
/* 53 */ { 5, s_5_53, 52, 5, 0},
/* 54 */ { 6, s_5_54, 53, 4, 0},
/* 55 */ { 5, s_5_55, 52, 3, 0},
/* 56 */ { 6, s_5_56, 55, 2, 0},
/* 57 */ { 6, s_5_57, 55, 1, 0},
/* 58 */ { 5, s_5_58, 52, 6, 0},
/* 59 */ { 4, s_5_59, -1, 8, 0},
/* 60 */ { 3, s_5_60, -1, 11, 0},
/* 61 */ { 2, s_5_61, -1, 10, 0},
/* 62 */ { 3, s_5_62, 61, 5, 0},
/* 63 */ { 4, s_5_63, 62, 4, 0},
/* 64 */ { 3, s_5_64, 61, 3, 0},
/* 65 */ { 4, s_5_65, 64, 2, 0},
/* 66 */ { 4, s_5_66, 64, 1, 0},
/* 67 */ { 4, s_5_67, 61, 11, 0},
/* 68 */ { 3, s_5_68, 61, 6, 0},
/* 69 */ { 4, s_5_69, 61, 9, 0},
/* 70 */ { 3, s_5_70, -1, 9, 0},
/* 71 */ { 3, s_5_71, -1, 9, 0},
/* 72 */ { 3, s_5_72, -1, 11, 0},
/* 73 */ { 3, s_5_73, -1, 9, 0},
/* 74 */ { 4, s_5_74, -1, 11, 0},
/* 75 */ { 4, s_5_75, -1, 9, 0},
/* 76 */ { 2, s_5_76, -1, 10, 0},
/* 77 */ { 3, s_5_77, 76, 5, 0},
/* 78 */ { 4, s_5_78, 77, 4, 0},
/* 79 */ { 3, s_5_79, 76, 3, 0},
/* 80 */ { 4, s_5_80, 79, 2, 0},
/* 81 */ { 4, s_5_81, 79, 1, 0},
/* 82 */ { 3, s_5_82, 76, 6, 0},
/* 83 */ { 2, s_5_83, -1, 10, 0},
/* 84 */ { 3, s_5_84, 83, 5, 0},
/* 85 */ { 4, s_5_85, 84, 4, 0},
/* 86 */ { 3, s_5_86, 83, 3, 0},
/* 87 */ { 4, s_5_87, 86, 2, 0},
/* 88 */ { 4, s_5_88, 86, 1, 0},
/* 89 */ { 3, s_5_89, 83, 6, 0},
/* 90 */ { 2, s_5_90, -1, 12, 0},
/* 91 */ { 3, s_5_91, 90, 3, 0},
/* 92 */ { 5, s_5_92, 90, 2, 0},
/* 93 */ { 5, s_5_93, 90, 1, 0},
/* 94 */ { 4, s_5_94, 90, 7, 0},
/* 95 */ { 4, s_5_95, 90, 6, 0},
/* 96 */ { 6, s_5_96, 90, 10, 0},
/* 97 */ { 8, s_5_97, 96, 6, 0},
/* 98 */ { 7, s_5_98, 90, 10, 0},
/* 99 */ { 10, s_5_99, 98, 2, 0},
/*100 */ { 10, s_5_100, 98, 1, 0},
/*101 */ { 5, s_5_101, 90, 6, 0},
/*102 */ { 6, s_5_102, 90, 4, 0},
/*103 */ { 6, s_5_103, 90, 5, 0},
/*104 */ { 2, s_5_104, -1, 10, 0},
/*105 */ { 3, s_5_105, 104, 5, 0},
/*106 */ { 4, s_5_106, 105, 4, 0},
/*107 */ { 3, s_5_107, 104, 3, 0},
/*108 */ { 4, s_5_108, 107, 2, 0},
/*109 */ { 4, s_5_109, 107, 1, 0},
/*110 */ { 3, s_5_110, 104, 6, 0}
};

static const symbol s_6_0[1] = { 't' };
static const symbol s_6_1[2] = { 'a', 't' };
static const symbol s_6_2[4] = { 'o', 'v', 'a', 't' };
static const symbol s_6_3[3] = { 'z', 'a', 't' };
static const symbol s_6_4[2] = { 'c', 't' };
static const symbol s_6_5[2] = { 'e', 't' };
static const symbol s_6_6[2] = { 'i', 't' };
static const symbol s_6_7[3] = { 'd', 'i', 't' };
static const symbol s_6_8[3] = { 's', 'i', 't' };
static const symbol s_6_9[4] = { 's', 't', 'i', 't' };
static const symbol s_6_10[4] = { 'n', 'o', 'u', 't' };
static const symbol s_6_11[6] = { 's', 'k', 'n', 'o', 'u', 't' };
static const symbol s_6_12[3] = { 0xC4, 0x9B, 't' };
static const symbol s_6_13[4] = { 'r', 0xC3, 0xA1, 't' };
static const symbol s_6_14[5] = { 0xC5, 0x99, 0xC3, 0xAD, 't' };

static const struct among a_6[15] =
{
/*  0 */ { 1, s_6_0, -1, 1, 0},
/*  1 */ { 2, s_6_1, 0, 1, 0},
/*  2 */ { 4, s_6_2, 1, 1, 0},
/*  3 */ { 3, s_6_3, 1, 1, 0},
/*  4 */ { 2, s_6_4, 0, 1, 0},
/*  5 */ { 2, s_6_5, 0, 1, 0},
/*  6 */ { 2, s_6_6, 0, 1, 0},
/*  7 */ { 3, s_6_7, 6, 1, 0},
/*  8 */ { 3, s_6_8, 6, 1, 0},
/*  9 */ { 4, s_6_9, 6, 1, 0},
/* 10 */ { 4, s_6_10, 0, 1, 0},
/* 11 */ { 6, s_6_11, 10, 1, 0},
/* 12 */ { 3, s_6_12, 0, 1, 0},
/* 13 */ { 4, s_6_13, 0, 1, 0},
/* 14 */ { 5, s_6_14, 0, 1, 0}
};

static const symbol s_7_0[1] = { 'a' };
static const symbol s_7_1[2] = { 'k', 'a' };
static const symbol s_7_2[2] = { 'l', 'a' };
static const symbol s_7_3[3] = { 'a', 'l', 'a' };
static const symbol s_7_4[4] = { 'r', 'a', 'l', 'a' };
static const symbol s_7_5[5] = { 'o', 'v', 'a', 'l', 'a' };
static const symbol s_7_6[4] = { 'z', 'a', 'l', 'a' };
static const symbol s_7_7[3] = { 'e', 'l', 'a' };
static const symbol s_7_8[5] = { 0xC5, 0x99, 'e', 'l', 'a' };
static const symbol s_7_9[3] = { 'i', 'l', 'a' };
static const symbol s_7_10[4] = { 'd', 'i', 'l', 'a' };
static const symbol s_7_11[4] = { 's', 'i', 'l', 'a' };
static const symbol s_7_12[5] = { 's', 't', 'i', 'l', 'a' };
static const symbol s_7_13[4] = { 'z', 'i', 'l', 'a' };
static const symbol s_7_14[3] = { 'k', 'l', 'a' };
static const symbol s_7_15[4] = { 's', 'k', 'l', 'a' };
static const symbol s_7_16[4] = { 'n', 'u', 'l', 'a' };
static const symbol s_7_17[4] = { 0xC4, 0x9B, 'l', 'a' };
static const symbol s_7_18[2] = { 'n', 'a' };
static const symbol s_7_19[3] = { 'e', 'n', 'a' };
static const symbol s_7_20[4] = { 'z', 'e', 'n', 'a' };
static const symbol s_7_21[5] = { 0xC4, 0x8D, 'e', 'n', 'a' };
static const symbol s_7_22[5] = { 0xC5, 0x99, 'e', 'n', 'a' };
static const symbol s_7_23[5] = { 0xC5, 0xA1, 'e', 'n', 'a' };
static const symbol s_7_24[5] = { 0xC5, 0xBE, 'e', 'n', 'a' };
static const symbol s_7_25[4] = { 's', 'k', 'n', 'a' };
static const symbol s_7_26[4] = { 0xC4, 0x9B, 'n', 'a' };
static const symbol s_7_27[6] = { 's', 't', 0xC4, 0x9B, 'n', 'a' };
static const symbol s_7_28[7] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'n', 'a' };
static const symbol s_7_29[4] = { 0xC3, 0xA1, 'n', 'a' };
static const symbol s_7_30[5] = { 'r', 0xC3, 0xA1, 'n', 'a' };
static const symbol s_7_31[6] = { 'o', 'v', 0xC3, 0xA1, 'n', 'a' };
static const symbol s_7_32[5] = { 'z', 0xC3, 0xA1, 'n', 'a' };
static const symbol s_7_33[2] = { 'r', 'a' };
static const symbol s_7_34[3] = { 'e', 'r', 'a' };
static const symbol s_7_35[2] = { 't', 'a' };
static const symbol s_7_36[3] = { 'a', 't', 'a' };
static const symbol s_7_37[4] = { 'n', 'u', 't', 'a' };
static const symbol s_7_38[6] = { 's', 'k', 'n', 'u', 't', 'a' };
static const symbol s_7_39[3] = { 'o', 'u', 'c' };
static const symbol s_7_40[4] = { 'k', 'o', 'u', 'c' };
static const symbol s_7_41[4] = { 'n', 'o', 'u', 'c' };
static const symbol s_7_42[6] = { 's', 'k', 'n', 'o', 'u', 'c' };
static const symbol s_7_43[4] = { 'r', 'o', 'u', 'c' };
static const symbol s_7_44[5] = { 'e', 'r', 'o', 'u', 'c' };
static const symbol s_7_45[3] = { 0xC3, 0xAD, 'c' };
static const symbol s_7_46[4] = { 'd', 0xC3, 0xAD, 'c' };
static const symbol s_7_47[4] = { 'j', 0xC3, 0xAD, 'c' };
static const symbol s_7_48[5] = { 'a', 'j', 0xC3, 0xAD, 'c' };
static const symbol s_7_49[5] = { 'e', 'j', 0xC3, 0xAD, 'c' };
static const symbol s_7_50[5] = { 'u', 'j', 0xC3, 0xAD, 'c' };
static const symbol s_7_51[4] = { 's', 0xC3, 0xAD, 'c' };
static const symbol s_7_52[5] = { 's', 't', 0xC3, 0xAD, 'c' };
static const symbol s_7_53[4] = { 'z', 0xC3, 0xAD, 'c' };
static const symbol s_7_54[5] = { 0xC5, 0xBE, 0xC3, 0xAD, 'c' };
static const symbol s_7_55[4] = { 'o', 'u', 'c', 'e' };
static const symbol s_7_56[5] = { 'k', 'o', 'u', 'c', 'e' };
static const symbol s_7_57[5] = { 'n', 'o', 'u', 'c', 'e' };
static const symbol s_7_58[7] = { 's', 'k', 'n', 'o', 'u', 'c', 'e' };
static const symbol s_7_59[5] = { 'r', 'o', 'u', 'c', 'e' };
static const symbol s_7_60[6] = { 'e', 'r', 'o', 'u', 'c', 'e' };
static const symbol s_7_61[4] = { 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_62[5] = { 'd', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_63[5] = { 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_64[6] = { 'a', 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_65[6] = { 'e', 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_66[6] = { 'u', 'j', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_67[5] = { 's', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_68[6] = { 's', 't', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_69[5] = { 'z', 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_70[6] = { 0xC5, 0xBE, 0xC3, 0xAD, 'c', 'e' };
static const symbol s_7_71[2] = { 'j', 'e' };
static const symbol s_7_72[3] = { 'a', 'j', 'e' };
static const symbol s_7_73[3] = { 'e', 'j', 'e' };
static const symbol s_7_74[3] = { 'u', 'j', 'e' };
static const symbol s_7_75[2] = { 'm', 'e' };
static const symbol s_7_76[3] = { 'e', 'm', 'e' };
static const symbol s_7_77[4] = { 'j', 'e', 'm', 'e' };
static const symbol s_7_78[5] = { 'u', 'j', 'e', 'm', 'e' };
static const symbol s_7_79[4] = { 'n', 'e', 'm', 'e' };
static const symbol s_7_80[6] = { 's', 'k', 'n', 'e', 'm', 'e' };
static const symbol s_7_81[5] = { 'e', 'r', 'e', 'm', 'e' };
static const symbol s_7_82[5] = { 0xC4, 0x8D, 'e', 'm', 'e' };
static const symbol s_7_83[5] = { 0xC5, 0x99, 'e', 'm', 'e' };
static const symbol s_7_84[5] = { 0xC5, 0xBE, 'e', 'm', 'e' };
static const symbol s_7_85[3] = { 'j', 'm', 'e' };
static const symbol s_7_86[4] = { 'e', 'j', 'm', 'e' };
static const symbol s_7_87[4] = { 'u', 'j', 'm', 'e' };
static const symbol s_7_88[4] = { 'e', 'r', 'm', 'e' };
static const symbol s_7_89[3] = { 's', 'm', 'e' };
static const symbol s_7_90[3] = { 'z', 'm', 'e' };
static const symbol s_7_91[4] = { 0xC5, 0x88, 'm', 'e' };
static const symbol s_7_92[4] = { 0xC4, 0x8D, 'm', 'e' };
static const symbol s_7_93[4] = { 0xC4, 0x8F, 'm', 'e' };
static const symbol s_7_94[5] = { 'n', 0xC4, 0x9B, 'm', 'e' };
static const symbol s_7_95[7] = { 's', 'k', 'n', 0xC4, 0x9B, 'm', 'e' };
static const symbol s_7_96[6] = { 's', 't', 0xC4, 0x9B, 'm', 'e' };
static const symbol s_7_97[4] = { 0xC3, 0xA1, 'm', 'e' };
static const symbol s_7_98[4] = { 0xC3, 0xAD, 'm', 'e' };
static const symbol s_7_99[5] = { 'd', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_7_100[5] = { 's', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_7_101[6] = { 's', 't', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_7_102[5] = { 'z', 0xC3, 0xAD, 'm', 'e' };
static const symbol s_7_103[4] = { 0xC5, 0xBE, 'm', 'e' };
static const symbol s_7_104[2] = { 'n', 'e' };
static const symbol s_7_105[4] = { 's', 'k', 'n', 'e' };
static const symbol s_7_106[3] = { 'e', 'r', 'e' };
static const symbol s_7_107[2] = { 's', 'e' };
static const symbol s_7_108[2] = { 't', 'e' };
static const symbol s_7_109[3] = { 'e', 't', 'e' };
static const symbol s_7_110[4] = { 'j', 'e', 't', 'e' };
static const symbol s_7_111[5] = { 'u', 'j', 'e', 't', 'e' };
static const symbol s_7_112[4] = { 'n', 'e', 't', 'e' };
static const symbol s_7_113[6] = { 's', 'k', 'n', 'e', 't', 'e' };
static const symbol s_7_114[5] = { 'e', 'r', 'e', 't', 'e' };
static const symbol s_7_115[5] = { 0xC4, 0x8D, 'e', 't', 'e' };
static const symbol s_7_116[5] = { 0xC5, 0x99, 'e', 't', 'e' };
static const symbol s_7_117[5] = { 0xC5, 0xBE, 'e', 't', 'e' };
static const symbol s_7_118[3] = { 'j', 't', 'e' };
static const symbol s_7_119[4] = { 'e', 'j', 't', 'e' };
static const symbol s_7_120[4] = { 'u', 'j', 't', 'e' };
static const symbol s_7_121[4] = { 'e', 'r', 't', 'e' };
static const symbol s_7_122[3] = { 's', 't', 'e' };
static const symbol s_7_123[3] = { 'z', 't', 'e' };
static const symbol s_7_124[4] = { 0xC5, 0x88, 't', 'e' };
static const symbol s_7_125[4] = { 0xC4, 0x8D, 't', 'e' };
static const symbol s_7_126[4] = { 0xC4, 0x8F, 't', 'e' };
static const symbol s_7_127[5] = { 'n', 0xC4, 0x9B, 't', 'e' };
static const symbol s_7_128[7] = { 's', 'k', 'n', 0xC4, 0x9B, 't', 'e' };
static const symbol s_7_129[6] = { 's', 't', 0xC4, 0x9B, 't', 'e' };
static const symbol s_7_130[4] = { 0xC3, 0xA1, 't', 'e' };
static const symbol s_7_131[4] = { 0xC3, 0xAD, 't', 'e' };
static const symbol s_7_132[5] = { 'd', 0xC3, 0xAD, 't', 'e' };
static const symbol s_7_133[5] = { 's', 0xC3, 0xAD, 't', 'e' };
static const symbol s_7_134[6] = { 's', 't', 0xC3, 0xAD, 't', 'e' };
static const symbol s_7_135[5] = { 'z', 0xC3, 0xAD, 't', 'e' };
static const symbol s_7_136[4] = { 0xC5, 0xBE, 't', 'e' };
static const symbol s_7_137[2] = { 'z', 'e' };
static const symbol s_7_138[3] = { 0xC4, 0x8D, 'e' };
static const symbol s_7_139[3] = { 0xC5, 0x99, 'e' };
static const symbol s_7_140[5] = { 'a', 'v', 0xC5, 0xA1, 'e' };
static const symbol s_7_141[6] = { 'n', 'u', 'v', 0xC5, 0xA1, 'e' };
static const symbol s_7_142[3] = { 0xC5, 0xBE, 'e' };
static const symbol s_7_143[2] = { 'j', 'i' };
static const symbol s_7_144[3] = { 'u', 'j', 'i' };
static const symbol s_7_145[2] = { 'l', 'i' };
static const symbol s_7_146[3] = { 'a', 'l', 'i' };
static const symbol s_7_147[4] = { 'r', 'a', 'l', 'i' };
static const symbol s_7_148[5] = { 'o', 'v', 'a', 'l', 'i' };
static const symbol s_7_149[4] = { 'z', 'a', 'l', 'i' };
static const symbol s_7_150[3] = { 'e', 'l', 'i' };
static const symbol s_7_151[5] = { 0xC5, 0x99, 'e', 'l', 'i' };
static const symbol s_7_152[3] = { 'i', 'l', 'i' };
static const symbol s_7_153[4] = { 'd', 'i', 'l', 'i' };
static const symbol s_7_154[4] = { 's', 'i', 'l', 'i' };
static const symbol s_7_155[5] = { 's', 't', 'i', 'l', 'i' };
static const symbol s_7_156[4] = { 'z', 'i', 'l', 'i' };
static const symbol s_7_157[3] = { 'k', 'l', 'i' };
static const symbol s_7_158[4] = { 's', 'k', 'l', 'i' };
static const symbol s_7_159[4] = { 'n', 'u', 'l', 'i' };
static const symbol s_7_160[4] = { 0xC4, 0x9B, 'l', 'i' };
static const symbol s_7_161[2] = { 'n', 'i' };
static const symbol s_7_162[3] = { 'e', 'n', 'i' };
static const symbol s_7_163[4] = { 'z', 'e', 'n', 'i' };
static const symbol s_7_164[5] = { 0xC4, 0x8D, 'e', 'n', 'i' };
static const symbol s_7_165[5] = { 0xC5, 0x99, 'e', 'n', 'i' };
static const symbol s_7_166[5] = { 0xC5, 0xA1, 'e', 'n', 'i' };
static const symbol s_7_167[5] = { 0xC5, 0xBE, 'e', 'n', 'i' };
static const symbol s_7_168[4] = { 's', 'k', 'n', 'i' };
static const symbol s_7_169[4] = { 0xC4, 0x9B, 'n', 'i' };
static const symbol s_7_170[6] = { 's', 't', 0xC4, 0x9B, 'n', 'i' };
static const symbol s_7_171[7] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'n', 'i' };
static const symbol s_7_172[4] = { 0xC3, 0xA1, 'n', 'i' };
static const symbol s_7_173[5] = { 'r', 0xC3, 0xA1, 'n', 'i' };
static const symbol s_7_174[6] = { 'o', 'v', 0xC3, 0xA1, 'n', 'i' };
static const symbol s_7_175[5] = { 'z', 0xC3, 0xA1, 'n', 'i' };
static const symbol s_7_176[2] = { 't', 'i' };
static const symbol s_7_177[3] = { 'a', 't', 'i' };
static const symbol s_7_178[3] = { 's', 't', 'i' };
static const symbol s_7_179[4] = { 'n', 'u', 't', 'i' };
static const symbol s_7_180[6] = { 's', 'k', 'n', 'u', 't', 'i' };
static const symbol s_7_181[3] = { 0xC5, 0x99, 'i' };
static const symbol s_7_182[5] = { 'a', 'v', 0xC5, 0xA1, 'i' };
static const symbol s_7_183[6] = { 'n', 'u', 'v', 0xC5, 0xA1, 'i' };
static const symbol s_7_184[1] = { 'j' };
static const symbol s_7_185[2] = { 'e', 'j' };
static const symbol s_7_186[2] = { 'u', 'j' };
static const symbol s_7_187[1] = { 'l' };
static const symbol s_7_188[2] = { 'a', 'l' };
static const symbol s_7_189[3] = { 'r', 'a', 'l' };
static const symbol s_7_190[4] = { 'o', 'v', 'a', 'l' };
static const symbol s_7_191[3] = { 'z', 'a', 'l' };
static const symbol s_7_192[2] = { 'e', 'l' };
static const symbol s_7_193[4] = { 0xC5, 0x99, 'e', 'l' };
static const symbol s_7_194[2] = { 'i', 'l' };
static const symbol s_7_195[3] = { 'd', 'i', 'l' };
static const symbol s_7_196[3] = { 's', 'i', 'l' };
static const symbol s_7_197[4] = { 's', 't', 'i', 'l' };
static const symbol s_7_198[3] = { 'z', 'i', 'l' };
static const symbol s_7_199[2] = { 'k', 'l' };
static const symbol s_7_200[3] = { 's', 'k', 'l' };
static const symbol s_7_201[3] = { 'n', 'u', 'l' };
static const symbol s_7_202[3] = { 0xC4, 0x9B, 'l' };
static const symbol s_7_203[3] = { 0xC3, 0xA1, 'm' };
static const symbol s_7_204[3] = { 0xC3, 0xAD, 'm' };
static const symbol s_7_205[4] = { 'd', 0xC3, 0xAD, 'm' };
static const symbol s_7_206[4] = { 's', 0xC3, 0xAD, 'm' };
static const symbol s_7_207[5] = { 's', 't', 0xC3, 0xAD, 'm' };
static const symbol s_7_208[4] = { 'z', 0xC3, 0xAD, 'm' };
static const symbol s_7_209[2] = { 'e', 'n' };
static const symbol s_7_210[3] = { 'z', 'e', 'n' };
static const symbol s_7_211[4] = { 0xC4, 0x8D, 'e', 'n' };
static const symbol s_7_212[4] = { 0xC5, 0x99, 'e', 'n' };
static const symbol s_7_213[4] = { 0xC5, 0xA1, 'e', 'n' };
static const symbol s_7_214[4] = { 0xC5, 0xBE, 'e', 'n' };
static const symbol s_7_215[3] = { 0xC4, 0x9B, 'n' };
static const symbol s_7_216[5] = { 's', 't', 0xC4, 0x9B, 'n' };
static const symbol s_7_217[6] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'n' };
static const symbol s_7_218[3] = { 0xC3, 0xA1, 'n' };
static const symbol s_7_219[4] = { 'r', 0xC3, 0xA1, 'n' };
static const symbol s_7_220[5] = { 'o', 'v', 0xC3, 0xA1, 'n' };
static const symbol s_7_221[4] = { 'z', 0xC3, 0xA1, 'n' };
static const symbol s_7_222[2] = { 'l', 'o' };
static const symbol s_7_223[3] = { 'a', 'l', 'o' };
static const symbol s_7_224[4] = { 'r', 'a', 'l', 'o' };
static const symbol s_7_225[5] = { 'o', 'v', 'a', 'l', 'o' };
static const symbol s_7_226[4] = { 'z', 'a', 'l', 'o' };
static const symbol s_7_227[3] = { 'e', 'l', 'o' };
static const symbol s_7_228[5] = { 0xC5, 0x99, 'e', 'l', 'o' };
static const symbol s_7_229[3] = { 'i', 'l', 'o' };
static const symbol s_7_230[4] = { 'd', 'i', 'l', 'o' };
static const symbol s_7_231[4] = { 's', 'i', 'l', 'o' };
static const symbol s_7_232[5] = { 's', 't', 'i', 'l', 'o' };
static const symbol s_7_233[4] = { 'z', 'i', 'l', 'o' };
static const symbol s_7_234[3] = { 'k', 'l', 'o' };
static const symbol s_7_235[4] = { 's', 'k', 'l', 'o' };
static const symbol s_7_236[4] = { 'n', 'u', 'l', 'o' };
static const symbol s_7_237[4] = { 0xC4, 0x9B, 'l', 'o' };
static const symbol s_7_238[3] = { 'e', 'n', 'o' };
static const symbol s_7_239[4] = { 'z', 'e', 'n', 'o' };
static const symbol s_7_240[5] = { 0xC4, 0x8D, 'e', 'n', 'o' };
static const symbol s_7_241[5] = { 0xC5, 0x99, 'e', 'n', 'o' };
static const symbol s_7_242[5] = { 0xC5, 0xA1, 'e', 'n', 'o' };
static const symbol s_7_243[5] = { 0xC5, 0xBE, 'e', 'n', 'o' };
static const symbol s_7_244[4] = { 0xC4, 0x9B, 'n', 'o' };
static const symbol s_7_245[6] = { 's', 't', 0xC4, 0x9B, 'n', 'o' };
static const symbol s_7_246[7] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'n', 'o' };
static const symbol s_7_247[4] = { 0xC3, 0xA1, 'n', 'o' };
static const symbol s_7_248[5] = { 'r', 0xC3, 0xA1, 'n', 'o' };
static const symbol s_7_249[6] = { 'o', 'v', 0xC3, 0xA1, 'n', 'o' };
static const symbol s_7_250[5] = { 'z', 0xC3, 0xA1, 'n', 'o' };
static const symbol s_7_251[2] = { 't', 'o' };
static const symbol s_7_252[3] = { 'a', 't', 'o' };
static const symbol s_7_253[4] = { 'n', 'u', 't', 'o' };
static const symbol s_7_254[6] = { 's', 'k', 'n', 'u', 't', 'o' };
static const symbol s_7_255[2] = { 'e', 'r' };
static const symbol s_7_256[1] = { 's' };
static const symbol s_7_257[2] = { 'a', 't' };
static const symbol s_7_258[3] = { 'n', 'u', 't' };
static const symbol s_7_259[5] = { 's', 'k', 'n', 'u', 't' };
static const symbol s_7_260[1] = { 'u' };
static const symbol s_7_261[2] = { 'j', 'u' };
static const symbol s_7_262[3] = { 'u', 'j', 'u' };
static const symbol s_7_263[2] = { 'n', 'u' };
static const symbol s_7_264[4] = { 's', 'k', 'n', 'u' };
static const symbol s_7_265[2] = { 'o', 'u' };
static const symbol s_7_266[3] = { 'j', 'o', 'u' };
static const symbol s_7_267[4] = { 'u', 'j', 'o', 'u' };
static const symbol s_7_268[3] = { 'n', 'o', 'u' };
static const symbol s_7_269[5] = { 's', 'k', 'n', 'o', 'u' };
static const symbol s_7_270[4] = { 'e', 'r', 'o', 'u' };
static const symbol s_7_271[4] = { 0xC4, 0x8D, 'o', 'u' };
static const symbol s_7_272[4] = { 0xC5, 0x99, 'o', 'u' };
static const symbol s_7_273[4] = { 0xC5, 0xBE, 'o', 'u' };
static const symbol s_7_274[3] = { 'e', 'r', 'u' };
static const symbol s_7_275[3] = { 0xC4, 0x8D, 'u' };
static const symbol s_7_276[3] = { 0xC5, 0x99, 'u' };
static const symbol s_7_277[3] = { 0xC5, 0xBE, 'u' };
static const symbol s_7_278[2] = { 'a', 'v' };
static const symbol s_7_279[3] = { 'n', 'u', 'v' };
static const symbol s_7_280[2] = { 'l', 'y' };
static const symbol s_7_281[3] = { 'a', 'l', 'y' };
static const symbol s_7_282[4] = { 'r', 'a', 'l', 'y' };
static const symbol s_7_283[5] = { 'o', 'v', 'a', 'l', 'y' };
static const symbol s_7_284[4] = { 'z', 'a', 'l', 'y' };
static const symbol s_7_285[3] = { 'e', 'l', 'y' };
static const symbol s_7_286[5] = { 0xC5, 0x99, 'e', 'l', 'y' };
static const symbol s_7_287[3] = { 'i', 'l', 'y' };
static const symbol s_7_288[4] = { 'd', 'i', 'l', 'y' };
static const symbol s_7_289[4] = { 's', 'i', 'l', 'y' };
static const symbol s_7_290[5] = { 's', 't', 'i', 'l', 'y' };
static const symbol s_7_291[4] = { 'z', 'i', 'l', 'y' };
static const symbol s_7_292[3] = { 'k', 'l', 'y' };
static const symbol s_7_293[4] = { 's', 'k', 'l', 'y' };
static const symbol s_7_294[4] = { 'n', 'u', 'l', 'y' };
static const symbol s_7_295[4] = { 0xC4, 0x9B, 'l', 'y' };
static const symbol s_7_296[3] = { 'e', 'n', 'y' };
static const symbol s_7_297[4] = { 'z', 'e', 'n', 'y' };
static const symbol s_7_298[5] = { 0xC4, 0x8D, 'e', 'n', 'y' };
static const symbol s_7_299[5] = { 0xC5, 0x99, 'e', 'n', 'y' };
static const symbol s_7_300[5] = { 0xC5, 0xA1, 'e', 'n', 'y' };
static const symbol s_7_301[5] = { 0xC5, 0xBE, 'e', 'n', 'y' };
static const symbol s_7_302[4] = { 0xC4, 0x9B, 'n', 'y' };
static const symbol s_7_303[6] = { 's', 't', 0xC4, 0x9B, 'n', 'y' };
static const symbol s_7_304[7] = { 0xC5, 0xA1, 't', 0xC4, 0x9B, 'n', 'y' };
static const symbol s_7_305[4] = { 0xC3, 0xA1, 'n', 'y' };
static const symbol s_7_306[5] = { 'r', 0xC3, 0xA1, 'n', 'y' };
static const symbol s_7_307[6] = { 'o', 'v', 0xC3, 0xA1, 'n', 'y' };
static const symbol s_7_308[5] = { 'z', 0xC3, 0xA1, 'n', 'y' };
static const symbol s_7_309[2] = { 't', 'y' };
static const symbol s_7_310[3] = { 'a', 't', 'y' };
static const symbol s_7_311[4] = { 'n', 'u', 't', 'y' };
static const symbol s_7_312[6] = { 's', 'k', 'n', 'u', 't', 'y' };
static const symbol s_7_313[1] = { 'z' };
static const symbol s_7_314[2] = { 0xC5, 0x88 };
static const symbol s_7_315[2] = { 0xC4, 0x8D };
static const symbol s_7_316[2] = { 0xC4, 0x8F };
static const symbol s_7_317[2] = { 0xC4, 0x9B };
static const symbol s_7_318[3] = { 'd', 0xC4, 0x9B };
static const symbol s_7_319[4] = { 's', 't', 0xC4, 0x9B };
static const symbol s_7_320[2] = { 0xC3, 0xA1 };
static const symbol s_7_321[3] = { 'e', 0xC5, 0xA1 };
static const symbol s_7_322[4] = { 'j', 'e', 0xC5, 0xA1 };
static const symbol s_7_323[5] = { 'u', 'j', 'e', 0xC5, 0xA1 };
static const symbol s_7_324[4] = { 'n', 'e', 0xC5, 0xA1 };
static const symbol s_7_325[6] = { 's', 'k', 'n', 'e', 0xC5, 0xA1 };
static const symbol s_7_326[5] = { 'e', 'r', 'e', 0xC5, 0xA1 };
static const symbol s_7_327[5] = { 0xC4, 0x8D, 'e', 0xC5, 0xA1 };
static const symbol s_7_328[5] = { 0xC5, 0x99, 'e', 0xC5, 0xA1 };
static const symbol s_7_329[5] = { 0xC5, 0xBE, 'e', 0xC5, 0xA1 };
static const symbol s_7_330[4] = { 0xC3, 0xA1, 0xC5, 0xA1 };
static const symbol s_7_331[4] = { 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_7_332[5] = { 'd', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_7_333[5] = { 's', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_7_334[6] = { 's', 't', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_7_335[5] = { 'z', 0xC3, 0xAD, 0xC5, 0xA1 };
static const symbol s_7_336[3] = { 'd', 0xC3, 0xAD };
static const symbol s_7_337[3] = { 'j', 0xC3, 0xAD };
static const symbol s_7_338[4] = { 'a', 'j', 0xC3, 0xAD };
static const symbol s_7_339[4] = { 'e', 'j', 0xC3, 0xAD };
static const symbol s_7_340[4] = { 'u', 'j', 0xC3, 0xAD };
static const symbol s_7_341[3] = { 's', 0xC3, 0xAD };
static const symbol s_7_342[4] = { 's', 't', 0xC3, 0xAD };
static const symbol s_7_343[3] = { 'z', 0xC3, 0xAD };
static const symbol s_7_344[2] = { 0xC5, 0xBE };

static const struct among a_7[345] =
{
/*  0 */ { 1, s_7_0, -1, 3, 0},
/*  1 */ { 2, s_7_1, 0, 6, 0},
/*  2 */ { 2, s_7_2, 0, 3, 0},
/*  3 */ { 3, s_7_3, 2, 16, 0},
/*  4 */ { 4, s_7_4, 3, 2, 0},
/*  5 */ { 5, s_7_5, 3, 1, 0},
/*  6 */ { 4, s_7_6, 3, 4, 0},
/*  7 */ { 3, s_7_7, 2, 8, 0},
/*  8 */ { 5, s_7_8, 7, 11, 0},
/*  9 */ { 3, s_7_9, 2, 13, 0},
/* 10 */ { 4, s_7_10, 9, 15, 0},
/* 11 */ { 4, s_7_11, 9, 7, 0},
/* 12 */ { 5, s_7_12, 9, 14, 0},
/* 13 */ { 4, s_7_13, 9, 13, 0},
/* 14 */ { 3, s_7_14, 2, 6, 0},
/* 15 */ { 4, s_7_15, 14, 9, 0},
/* 16 */ { 4, s_7_16, 2, 5, 0},
/* 17 */ { 4, s_7_17, 2, 10, 0},
/* 18 */ { 2, s_7_18, 0, 5, 0},
/* 19 */ { 3, s_7_19, 18, 16, 0},
/* 20 */ { 4, s_7_20, 19, 15, 0},
/* 21 */ { 5, s_7_21, 19, 6, 0},
/* 22 */ { 5, s_7_22, 19, 11, 0},
/* 23 */ { 5, s_7_23, 19, 7, 0},
/* 24 */ { 5, s_7_24, 19, 13, 0},
/* 25 */ { 4, s_7_25, 18, 9, 0},
/* 26 */ { 4, s_7_26, 18, 10, 0},
/* 27 */ { 6, s_7_27, 26, 14, 0},
/* 28 */ { 7, s_7_28, 26, 16, 0},
/* 29 */ { 4, s_7_29, 18, 12, 0},
/* 30 */ { 5, s_7_30, 29, 2, 0},
/* 31 */ { 6, s_7_31, 29, 1, 0},
/* 32 */ { 5, s_7_32, 29, 4, 0},
/* 33 */ { 2, s_7_33, 0, 11, 0},
/* 34 */ { 3, s_7_34, 33, 2, 0},
/* 35 */ { 2, s_7_35, 0, 3, 0},
/* 36 */ { 3, s_7_36, 35, 5, 0},
/* 37 */ { 4, s_7_37, 35, 5, 0},
/* 38 */ { 6, s_7_38, 37, 9, 0},
/* 39 */ { 3, s_7_39, -1, 3, 0},
/* 40 */ { 4, s_7_40, 39, 6, 0},
/* 41 */ { 4, s_7_41, 39, 5, 0},
/* 42 */ { 6, s_7_42, 41, 9, 0},
/* 43 */ { 4, s_7_43, 39, 11, 0},
/* 44 */ { 5, s_7_44, 43, 2, 0},
/* 45 */ { 3, s_7_45, -1, 16, 0},
/* 46 */ { 4, s_7_46, 45, 15, 0},
/* 47 */ { 4, s_7_47, 45, 3, 0},
/* 48 */ { 5, s_7_48, 47, 12, 0},
/* 49 */ { 5, s_7_49, 47, 8, 0},
/* 50 */ { 5, s_7_50, 47, 1, 0},
/* 51 */ { 4, s_7_51, 45, 7, 0},
/* 52 */ { 5, s_7_52, 45, 14, 0},
/* 53 */ { 4, s_7_53, 45, 13, 0},
/* 54 */ { 5, s_7_54, 45, 4, 0},
/* 55 */ { 4, s_7_55, -1, 3, 0},
/* 56 */ { 5, s_7_56, 55, 6, 0},
/* 57 */ { 5, s_7_57, 55, 5, 0},
/* 58 */ { 7, s_7_58, 57, 9, 0},
/* 59 */ { 5, s_7_59, 55, 11, 0},
/* 60 */ { 6, s_7_60, 59, 2, 0},
/* 61 */ { 4, s_7_61, -1, 16, 0},
/* 62 */ { 5, s_7_62, 61, 15, 0},
/* 63 */ { 5, s_7_63, 61, 3, 0},
/* 64 */ { 6, s_7_64, 63, 12, 0},
/* 65 */ { 6, s_7_65, 63, 8, 0},
/* 66 */ { 6, s_7_66, 63, 1, 0},
/* 67 */ { 5, s_7_67, 61, 7, 0},
/* 68 */ { 6, s_7_68, 61, 14, 0},
/* 69 */ { 5, s_7_69, 61, 13, 0},
/* 70 */ { 6, s_7_70, 61, 4, 0},
/* 71 */ { 2, s_7_71, -1, 3, 0},
/* 72 */ { 3, s_7_72, 71, 12, 0},
/* 73 */ { 3, s_7_73, 71, 8, 0},
/* 74 */ { 3, s_7_74, 71, 1, 0},
/* 75 */ { 2, s_7_75, -1, 16, 0},
/* 76 */ { 3, s_7_76, 75, 3, 0},
/* 77 */ { 4, s_7_77, 76, 3, 0},
/* 78 */ { 5, s_7_78, 77, 1, 0},
/* 79 */ { 4, s_7_79, 76, 5, 0},
/* 80 */ { 6, s_7_80, 79, 9, 0},
/* 81 */ { 5, s_7_81, 76, 2, 0},
/* 82 */ { 5, s_7_82, 76, 6, 0},
/* 83 */ { 5, s_7_83, 76, 11, 0},
/* 84 */ { 5, s_7_84, 76, 4, 0},
/* 85 */ { 3, s_7_85, 75, 3, 0},
/* 86 */ { 4, s_7_86, 85, 16, 0},
/* 87 */ { 4, s_7_87, 85, 1, 0},
/* 88 */ { 4, s_7_88, 75, 2, 0},
/* 89 */ { 3, s_7_89, 75, 7, 0},
/* 90 */ { 3, s_7_90, 75, 13, 0},
/* 91 */ { 4, s_7_91, 75, 5, 0},
/* 92 */ { 4, s_7_92, 75, 6, 0},
/* 93 */ { 4, s_7_93, 75, 15, 0},
/* 94 */ { 5, s_7_94, 75, 5, 0},
/* 95 */ { 7, s_7_95, 94, 9, 0},
/* 96 */ { 6, s_7_96, 75, 14, 0},
/* 97 */ { 4, s_7_97, 75, 12, 0},
/* 98 */ { 4, s_7_98, 75, 16, 0},
/* 99 */ { 5, s_7_99, 98, 15, 0},
/*100 */ { 5, s_7_100, 98, 7, 0},
/*101 */ { 6, s_7_101, 98, 14, 0},
/*102 */ { 5, s_7_102, 98, 13, 0},
/*103 */ { 4, s_7_103, 75, 4, 0},
/*104 */ { 2, s_7_104, -1, 5, 0},
/*105 */ { 4, s_7_105, 104, 9, 0},
/*106 */ { 3, s_7_106, -1, 2, 0},
/*107 */ { 2, s_7_107, -1, 7, 0},
/*108 */ { 2, s_7_108, -1, 16, 0},
/*109 */ { 3, s_7_109, 108, 3, 0},
/*110 */ { 4, s_7_110, 109, 3, 0},
/*111 */ { 5, s_7_111, 110, 1, 0},
/*112 */ { 4, s_7_112, 109, 5, 0},
/*113 */ { 6, s_7_113, 112, 9, 0},
/*114 */ { 5, s_7_114, 109, 2, 0},
/*115 */ { 5, s_7_115, 109, 6, 0},
/*116 */ { 5, s_7_116, 109, 11, 0},
/*117 */ { 5, s_7_117, 109, 4, 0},
/*118 */ { 3, s_7_118, 108, 3, 0},
/*119 */ { 4, s_7_119, 118, 16, 0},
/*120 */ { 4, s_7_120, 118, 1, 0},
/*121 */ { 4, s_7_121, 108, 2, 0},
/*122 */ { 3, s_7_122, 108, 7, 0},
/*123 */ { 3, s_7_123, 108, 13, 0},
/*124 */ { 4, s_7_124, 108, 5, 0},
/*125 */ { 4, s_7_125, 108, 6, 0},
/*126 */ { 4, s_7_126, 108, 15, 0},
/*127 */ { 5, s_7_127, 108, 5, 0},
/*128 */ { 7, s_7_128, 127, 9, 0},
/*129 */ { 6, s_7_129, 108, 14, 0},
/*130 */ { 4, s_7_130, 108, 12, 0},
/*131 */ { 4, s_7_131, 108, 16, 0},
/*132 */ { 5, s_7_132, 131, 15, 0},
/*133 */ { 5, s_7_133, 131, 7, 0},
/*134 */ { 6, s_7_134, 131, 14, 0},
/*135 */ { 5, s_7_135, 131, 13, 0},
/*136 */ { 4, s_7_136, 108, 4, 0},
/*137 */ { 2, s_7_137, -1, 13, 0},
/*138 */ { 3, s_7_138, -1, 6, 0},
/*139 */ { 3, s_7_139, -1, 11, 0},
/*140 */ { 5, s_7_140, -1, 5, 0},
/*141 */ { 6, s_7_141, -1, 5, 0},
/*142 */ { 3, s_7_142, -1, 4, 0},
/*143 */ { 2, s_7_143, -1, 3, 0},
/*144 */ { 3, s_7_144, 143, 1, 0},
/*145 */ { 2, s_7_145, -1, 3, 0},
/*146 */ { 3, s_7_146, 145, 16, 0},
/*147 */ { 4, s_7_147, 146, 2, 0},
/*148 */ { 5, s_7_148, 146, 1, 0},
/*149 */ { 4, s_7_149, 146, 4, 0},
/*150 */ { 3, s_7_150, 145, 8, 0},
/*151 */ { 5, s_7_151, 150, 11, 0},
/*152 */ { 3, s_7_152, 145, 13, 0},
/*153 */ { 4, s_7_153, 152, 15, 0},
/*154 */ { 4, s_7_154, 152, 7, 0},
/*155 */ { 5, s_7_155, 152, 14, 0},
/*156 */ { 4, s_7_156, 152, 13, 0},
/*157 */ { 3, s_7_157, 145, 6, 0},
/*158 */ { 4, s_7_158, 157, 9, 0},
/*159 */ { 4, s_7_159, 145, 5, 0},
/*160 */ { 4, s_7_160, 145, 10, 0},
/*161 */ { 2, s_7_161, -1, 5, 0},
/*162 */ { 3, s_7_162, 161, 16, 0},
/*163 */ { 4, s_7_163, 162, 15, 0},
/*164 */ { 5, s_7_164, 162, 6, 0},
/*165 */ { 5, s_7_165, 162, 11, 0},
/*166 */ { 5, s_7_166, 162, 7, 0},
/*167 */ { 5, s_7_167, 162, 13, 0},
/*168 */ { 4, s_7_168, 161, 9, 0},
/*169 */ { 4, s_7_169, 161, 10, 0},
/*170 */ { 6, s_7_170, 169, 14, 0},
/*171 */ { 7, s_7_171, 169, 16, 0},
/*172 */ { 4, s_7_172, 161, 12, 0},
/*173 */ { 5, s_7_173, 172, 2, 0},
/*174 */ { 6, s_7_174, 172, 1, 0},
/*175 */ { 5, s_7_175, 172, 4, 0},
/*176 */ { 2, s_7_176, -1, 3, 0},
/*177 */ { 3, s_7_177, 176, 5, 0},
/*178 */ { 3, s_7_178, 176, 14, 0},
/*179 */ { 4, s_7_179, 176, 5, 0},
/*180 */ { 6, s_7_180, 179, 9, 0},
/*181 */ { 3, s_7_181, -1, 11, 0},
/*182 */ { 5, s_7_182, -1, 5, 0},
/*183 */ { 6, s_7_183, -1, 5, 0},
/*184 */ { 1, s_7_184, -1, 3, 0},
/*185 */ { 2, s_7_185, 184, 16, 0},
/*186 */ { 2, s_7_186, 184, 1, 0},
/*187 */ { 1, s_7_187, -1, 3, 0},
/*188 */ { 2, s_7_188, 187, 16, 0},
/*189 */ { 3, s_7_189, 188, 2, 0},
/*190 */ { 4, s_7_190, 188, 1, 0},
/*191 */ { 3, s_7_191, 188, 4, 0},
/*192 */ { 2, s_7_192, 187, 8, 0},
/*193 */ { 4, s_7_193, 192, 11, 0},
/*194 */ { 2, s_7_194, 187, 13, 0},
/*195 */ { 3, s_7_195, 194, 15, 0},
/*196 */ { 3, s_7_196, 194, 7, 0},
/*197 */ { 4, s_7_197, 194, 14, 0},
/*198 */ { 3, s_7_198, 194, 13, 0},
/*199 */ { 2, s_7_199, 187, 6, 0},
/*200 */ { 3, s_7_200, 199, 9, 0},
/*201 */ { 3, s_7_201, 187, 5, 0},
/*202 */ { 3, s_7_202, 187, 10, 0},
/*203 */ { 3, s_7_203, -1, 12, 0},
/*204 */ { 3, s_7_204, -1, 16, 0},
/*205 */ { 4, s_7_205, 204, 15, 0},
/*206 */ { 4, s_7_206, 204, 7, 0},
/*207 */ { 5, s_7_207, 204, 14, 0},
/*208 */ { 4, s_7_208, 204, 13, 0},
/*209 */ { 2, s_7_209, -1, 16, 0},
/*210 */ { 3, s_7_210, 209, 15, 0},
/*211 */ { 4, s_7_211, 209, 6, 0},
/*212 */ { 4, s_7_212, 209, 11, 0},
/*213 */ { 4, s_7_213, 209, 7, 0},
/*214 */ { 4, s_7_214, 209, 13, 0},
/*215 */ { 3, s_7_215, -1, 10, 0},
/*216 */ { 5, s_7_216, 215, 14, 0},
/*217 */ { 6, s_7_217, 215, 16, 0},
/*218 */ { 3, s_7_218, -1, 12, 0},
/*219 */ { 4, s_7_219, 218, 2, 0},
/*220 */ { 5, s_7_220, 218, 1, 0},
/*221 */ { 4, s_7_221, 218, 4, 0},
/*222 */ { 2, s_7_222, -1, 3, 0},
/*223 */ { 3, s_7_223, 222, 16, 0},
/*224 */ { 4, s_7_224, 223, 2, 0},
/*225 */ { 5, s_7_225, 223, 1, 0},
/*226 */ { 4, s_7_226, 223, 4, 0},
/*227 */ { 3, s_7_227, 222, 8, 0},
/*228 */ { 5, s_7_228, 227, 11, 0},
/*229 */ { 3, s_7_229, 222, 13, 0},
/*230 */ { 4, s_7_230, 229, 15, 0},
/*231 */ { 4, s_7_231, 229, 7, 0},
/*232 */ { 5, s_7_232, 229, 14, 0},
/*233 */ { 4, s_7_233, 229, 13, 0},
/*234 */ { 3, s_7_234, 222, 6, 0},
/*235 */ { 4, s_7_235, 234, 9, 0},
/*236 */ { 4, s_7_236, 222, 5, 0},
/*237 */ { 4, s_7_237, 222, 10, 0},
/*238 */ { 3, s_7_238, -1, 16, 0},
/*239 */ { 4, s_7_239, 238, 15, 0},
/*240 */ { 5, s_7_240, 238, 6, 0},
/*241 */ { 5, s_7_241, 238, 11, 0},
/*242 */ { 5, s_7_242, 238, 7, 0},
/*243 */ { 5, s_7_243, 238, 13, 0},
/*244 */ { 4, s_7_244, -1, 10, 0},
/*245 */ { 6, s_7_245, 244, 14, 0},
/*246 */ { 7, s_7_246, 244, 16, 0},
/*247 */ { 4, s_7_247, -1, 12, 0},
/*248 */ { 5, s_7_248, 247, 2, 0},
/*249 */ { 6, s_7_249, 247, 1, 0},
/*250 */ { 5, s_7_250, 247, 4, 0},
/*251 */ { 2, s_7_251, -1, 3, 0},
/*252 */ { 3, s_7_252, 251, 5, 0},
/*253 */ { 4, s_7_253, 251, 5, 0},
/*254 */ { 6, s_7_254, 253, 9, 0},
/*255 */ { 2, s_7_255, -1, 2, 0},
/*256 */ { 1, s_7_256, -1, 7, 0},
/*257 */ { 2, s_7_257, -1, 5, 0},
/*258 */ { 3, s_7_258, -1, 5, 0},
/*259 */ { 5, s_7_259, 258, 9, 0},
/*260 */ { 1, s_7_260, -1, 3, 0},
/*261 */ { 2, s_7_261, 260, 3, 0},
/*262 */ { 3, s_7_262, 261, 1, 0},
/*263 */ { 2, s_7_263, 260, 5, 0},
/*264 */ { 4, s_7_264, 263, 9, 0},
/*265 */ { 2, s_7_265, 260, 3, 0},
/*266 */ { 3, s_7_266, 265, 3, 0},
/*267 */ { 4, s_7_267, 266, 1, 0},
/*268 */ { 3, s_7_268, 265, 5, 0},
/*269 */ { 5, s_7_269, 268, 9, 0},
/*270 */ { 4, s_7_270, 265, 2, 0},
/*271 */ { 4, s_7_271, 265, 6, 0},
/*272 */ { 4, s_7_272, 265, 11, 0},
/*273 */ { 4, s_7_273, 265, 4, 0},
/*274 */ { 3, s_7_274, 260, 2, 0},
/*275 */ { 3, s_7_275, 260, 6, 0},
/*276 */ { 3, s_7_276, 260, 11, 0},
/*277 */ { 3, s_7_277, 260, 4, 0},
/*278 */ { 2, s_7_278, -1, 5, 0},
/*279 */ { 3, s_7_279, -1, 5, 0},
/*280 */ { 2, s_7_280, -1, 3, 0},
/*281 */ { 3, s_7_281, 280, 16, 0},
/*282 */ { 4, s_7_282, 281, 2, 0},
/*283 */ { 5, s_7_283, 281, 1, 0},
/*284 */ { 4, s_7_284, 281, 4, 0},
/*285 */ { 3, s_7_285, 280, 8, 0},
/*286 */ { 5, s_7_286, 285, 11, 0},
/*287 */ { 3, s_7_287, 280, 13, 0},
/*288 */ { 4, s_7_288, 287, 15, 0},
/*289 */ { 4, s_7_289, 287, 7, 0},
/*290 */ { 5, s_7_290, 287, 14, 0},
/*291 */ { 4, s_7_291, 287, 13, 0},
/*292 */ { 3, s_7_292, 280, 6, 0},
/*293 */ { 4, s_7_293, 292, 9, 0},
/*294 */ { 4, s_7_294, 280, 5, 0},
/*295 */ { 4, s_7_295, 280, 10, 0},
/*296 */ { 3, s_7_296, -1, 16, 0},
/*297 */ { 4, s_7_297, 296, 15, 0},
/*298 */ { 5, s_7_298, 296, 6, 0},
/*299 */ { 5, s_7_299, 296, 11, 0},
/*300 */ { 5, s_7_300, 296, 7, 0},
/*301 */ { 5, s_7_301, 296, 13, 0},
/*302 */ { 4, s_7_302, -1, 10, 0},
/*303 */ { 6, s_7_303, 302, 14, 0},
/*304 */ { 7, s_7_304, 302, 16, 0},
/*305 */ { 4, s_7_305, -1, 12, 0},
/*306 */ { 5, s_7_306, 305, 2, 0},
/*307 */ { 6, s_7_307, 305, 1, 0},
/*308 */ { 5, s_7_308, 305, 4, 0},
/*309 */ { 2, s_7_309, -1, 3, 0},
/*310 */ { 3, s_7_310, 309, 5, 0},
/*311 */ { 4, s_7_311, 309, 5, 0},
/*312 */ { 6, s_7_312, 311, 9, 0},
/*313 */ { 1, s_7_313, -1, 13, 0},
/*314 */ { 2, s_7_314, -1, 5, 0},
/*315 */ { 2, s_7_315, -1, 6, 0},
/*316 */ { 2, s_7_316, -1, 15, 0},
/*317 */ { 2, s_7_317, -1, 10, 0},
/*318 */ { 3, s_7_318, 317, 15, 0},
/*319 */ { 4, s_7_319, 317, 14, 0},
/*320 */ { 2, s_7_320, -1, 12, 0},
/*321 */ { 3, s_7_321, -1, 3, 0},
/*322 */ { 4, s_7_322, 321, 3, 0},
/*323 */ { 5, s_7_323, 322, 1, 0},
/*324 */ { 4, s_7_324, 321, 5, 0},
/*325 */ { 6, s_7_325, 324, 9, 0},
/*326 */ { 5, s_7_326, 321, 2, 0},
/*327 */ { 5, s_7_327, 321, 6, 0},
/*328 */ { 5, s_7_328, 321, 11, 0},
/*329 */ { 5, s_7_329, 321, 4, 0},
/*330 */ { 4, s_7_330, -1, 12, 0},
/*331 */ { 4, s_7_331, -1, 16, 0},
/*332 */ { 5, s_7_332, 331, 15, 0},
/*333 */ { 5, s_7_333, 331, 7, 0},
/*334 */ { 6, s_7_334, 331, 14, 0},
/*335 */ { 5, s_7_335, 331, 13, 0},
/*336 */ { 3, s_7_336, -1, 15, 0},
/*337 */ { 3, s_7_337, -1, 3, 0},
/*338 */ { 4, s_7_338, 337, 12, 0},
/*339 */ { 4, s_7_339, 337, 8, 0},
/*340 */ { 4, s_7_340, 337, 1, 0},
/*341 */ { 3, s_7_341, -1, 7, 0},
/*342 */ { 4, s_7_342, -1, 14, 0},
/*343 */ { 3, s_7_343, -1, 13, 0},
/*344 */ { 2, s_7_344, -1, 4, 0}
};

static const symbol s_8_0[3] = { 'h', 'l', 'e' };
static const symbol s_8_1[4] = { 'k', 'o', 'l', 'i' };
static const symbol s_8_2[2] = { 's', 'i' };
static const symbol s_8_3[2] = { 'm', 'o' };
static const symbol s_8_4[5] = { 'k', 'o', 'l', 'i', 'v' };
static const symbol s_8_5[2] = { 'k', 'y' };

static const struct among a_8[6] =
{
/*  0 */ { 3, s_8_0, -1, 1, 0},
/*  1 */ { 4, s_8_1, -1, 1, 0},
/*  2 */ { 2, s_8_2, -1, 1, 0},
/*  3 */ { 2, s_8_3, -1, 1, 0},
/*  4 */ { 5, s_8_4, -1, 1, 0},
/*  5 */ { 2, s_8_5, -1, 1, 0}
};

static const symbol s_9_0[2] = { 'b', 'b' };
static const symbol s_9_1[2] = { 'd', 'd' };
static const symbol s_9_2[2] = { 'f', 'f' };
static const symbol s_9_3[2] = { 'g', 'g' };
static const symbol s_9_4[2] = { 'm', 'm' };
static const symbol s_9_5[2] = { 'n', 'n' };
static const symbol s_9_6[2] = { 'p', 'p' };
static const symbol s_9_7[2] = { 'r', 'r' };
static const symbol s_9_8[2] = { 't', 't' };

static const struct among a_9[9] =
{
/*  0 */ { 2, s_9_0, -1, 1, 0},
/*  1 */ { 2, s_9_1, -1, 1, 0},
/*  2 */ { 2, s_9_2, -1, 1, 0},
/*  3 */ { 2, s_9_3, -1, 1, 0},
/*  4 */ { 2, s_9_4, -1, 1, 0},
/*  5 */ { 2, s_9_5, -1, 1, 0},
/*  6 */ { 2, s_9_6, -1, 1, 0},
/*  7 */ { 2, s_9_7, -1, 1, 0},
/*  8 */ { 2, s_9_8, -1, 1, 0}
};

static const unsigned char g_vowel[] = { 17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 17, 4, 18, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64 };

static const unsigned char g_consonant[] = { 119, 223, 119, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 64, 0, 128, 128, 8, 0, 0, 16 };

static const unsigned char g_aeiou[] = { 17, 65, 16 };

static const symbol s_0[] = { 'a' };
static const symbol s_1[] = { 'e' };
static const symbol s_2[] = { 'i' };
static const symbol s_3[] = { 'o' };
static const symbol s_4[] = { 'u' };
static const symbol s_5[] = { 'y' };
static const symbol s_6[] = { 'c' };
static const symbol s_7[] = { 'd' };
static const symbol s_8[] = { 'e' };
static const symbol s_9[] = { 'n' };
static const symbol s_10[] = { 'r' };
static const symbol s_11[] = { 's' };
static const symbol s_12[] = { 't' };
static const symbol s_13[] = { 'u' };
static const symbol s_14[] = { 'z' };
static const symbol s_15[] = { 'b', 'y', 't' };
static const symbol s_16[] = { 'm', 'i', 't' };
static const symbol s_17[] = { 'j', 'i', 't' };
static const symbol s_18[] = { 'j', 'i', 's', 't' };
static const symbol s_19[] = { 's', 'n', 'i', 's', 't' };
static const symbol s_20[] = { 'v', 'i', 'd', 'e', 't' };
static const symbol s_21[] = { 'v', 'e', 'd', 'e', 't' };
static const symbol s_22[] = { 'c', 'h', 't', 'i', 't' };
static const symbol s_23[] = { 0xC5, 0xA1 };
static const symbol s_24[] = { 'c', 'h' };
static const symbol s_25[] = { 'b', 'a' };
static const symbol s_26[] = { 'e', 'c' };
static const symbol s_27[] = { 'l', 'o' };
static const symbol s_28[] = { 'r', 'a' };
static const symbol s_29[] = { 'n', 'a' };
static const symbol s_30[] = { 'e' };
static const symbol s_31[] = { 'n', 'o' };
static const symbol s_32[] = { 'v', 'a' };
static const symbol s_33[] = { 'e', 'k' };
static const symbol s_34[] = { 't', 0xC4, 0x9B };
static const symbol s_35[] = { 'g' };
static const symbol s_36[] = { 'v', 'o' };
static const symbol s_37[] = { 'a' };
static const symbol s_38[] = { 0xC5, 0x88 };
static const symbol s_39[] = { 'y' };
static const symbol s_40[] = { 'n', 0xC4, 0x9B };
static const symbol s_41[] = { 'e', 0xC5, 0x88 };
static const symbol s_42[] = { 0xC4, 0x9B };
static const symbol s_43[] = { 'k', 'a' };
static const symbol s_44[] = { 0xC5, 0xBE };
static const symbol s_45[] = { 'h' };
static const symbol s_46[] = { 0xC5, 0xA1 };
static const symbol s_47[] = { 'c', 'h' };
static const symbol s_48[] = { 0xC5, 0xBE };
static const symbol s_49[] = { 'h' };
static const symbol s_50[] = { 0xC5, 0xA1 };
static const symbol s_51[] = { 'c', 'h' };
static const symbol s_52[] = { 0xC4, 0x8D };
static const symbol s_53[] = { 'k' };
static const symbol s_54[] = { 0xC4, 0x8D };
static const symbol s_55[] = { 'k' };
static const symbol s_56[] = { 0xC4, 0x8D };
static const symbol s_57[] = { 'k' };
static const symbol s_58[] = { 0xC4, 0x8D, 0xC5, 0xA5 };
static const symbol s_59[] = { 'c', 'k' };
static const symbol s_60[] = { 0xC5, 0xA1, 0xC5, 0xA5 };
static const symbol s_61[] = { 's', 'k' };
static const symbol s_62[] = { 0xC5, 0xBE };
static const symbol s_63[] = { 'h' };
static const symbol s_64[] = { 0xC5, 0xA1 };
static const symbol s_65[] = { 'c', 'h' };
static const symbol s_66[] = { 's', 'k', 0xC3, 0xBD };
static const symbol s_67[] = { 'c', 'k', 0xC3, 0xBD };
static const symbol s_68[] = { 'k', 0xC3, 0xBD };
static const symbol s_69[] = { 'c', 'h', 0xC3, 0xBD };
static const symbol s_70[] = { 'h', 0xC3, 0xBD };
static const symbol s_71[] = { 'r', 0xC3, 0xBD };
static const symbol s_72[] = { 'a', 'v', 0xC3, 0xBD };
static const symbol s_73[] = { 0xC3, 0xAD };
static const symbol s_74[] = { 0xC5, 0xAF, 'v' };
static const symbol s_75[] = { 0xC3, 0xBD };
static const symbol s_76[] = { 'i', 'n' };
static const symbol s_77[] = { 0xC3, 0xBD };
static const symbol s_78[] = { 'o', 'v', 'a', 't' };
static const symbol s_79[] = { 'r', 0xC3, 0xA1, 't' };
static const symbol s_80[] = { 't' };
static const symbol s_81[] = { 'z', 'a', 't' };
static const symbol s_82[] = { 'n', 'o', 'u', 't' };
static const symbol s_83[] = { 'c', 't' };
static const symbol s_84[] = { 's', 'i', 't' };
static const symbol s_85[] = { 'e', 't' };
static const symbol s_86[] = { 's', 'k', 'n', 'o', 'u', 't' };
static const symbol s_87[] = { 0xC4, 0x9B, 't' };
static const symbol s_88[] = { 0xC5, 0x99, 0xC3, 0xAD, 't' };
static const symbol s_89[] = { 'a', 't' };
static const symbol s_90[] = { 'z', 'i', 't' };
static const symbol s_91[] = { 's', 't', 'i', 't' };
static const symbol s_92[] = { 'd', 'i', 't' };
static const symbol s_93[] = { 0xC4, 0x8D };
static const symbol s_94[] = { 'k' };
static const symbol s_95[] = { 0xC5, 0xA1 };
static const symbol s_96[] = { 'c', 'h' };

static int r_mark_regions(struct SN_env * z) {
    z->I[0] = z->l;
    {    /* gopast */ /* grouping vowel, line 49 */
        int ret = out_grouping_U(z, g_vowel, 97, 367, 1);
        if (ret < 0) return 0;
        z->c += ret;
    }
    {    /* gopast */ /* non vowel, line 49 */
        int ret = in_grouping_U(z, g_vowel, 97, 367, 1);
        if (ret < 0) return 0;
        z->c += ret;
    }
    z->I[0] = z->c; /* setmark p1, line 49 */
    return 1;
}

static int r_postlude(struct SN_env * z) {
    int among_var;
    while(1) { /* repeat, line 52 */
        int c1 = z->c;
        z->bra = z->c; /* [, line 53 */
        among_var = find_among(z, a_0, 16); /* substring, line 53 */
        if (!(among_var)) goto lab0;
        z->ket = z->c; /* ], line 53 */
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_from_s(z, 1, s_0); /* <-, line 54 */
                    if (ret < 0) return ret;
                }
                break;
            case 2:
                {   int ret = slice_from_s(z, 1, s_1); /* <-, line 55 */
                    if (ret < 0) return ret;
                }
                break;
            case 3:
                {   int ret = slice_from_s(z, 1, s_2); /* <-, line 56 */
                    if (ret < 0) return ret;
                }
                break;
            case 4:
                {   int ret = slice_from_s(z, 1, s_3); /* <-, line 57 */
                    if (ret < 0) return ret;
                }
                break;
            case 5:
                {   int ret = slice_from_s(z, 1, s_4); /* <-, line 58 */
                    if (ret < 0) return ret;
                }
                break;
            case 6:
                {   int ret = slice_from_s(z, 1, s_5); /* <-, line 59 */
                    if (ret < 0) return ret;
                }
                break;
            case 7:
                {   int ret = slice_from_s(z, 1, s_6); /* <-, line 60 */
                    if (ret < 0) return ret;
                }
                break;
            case 8:
                {   int ret = slice_from_s(z, 1, s_7); /* <-, line 61 */
                    if (ret < 0) return ret;
                }
                break;
            case 9:
                {   int ret = slice_from_s(z, 1, s_8); /* <-, line 62 */
                    if (ret < 0) return ret;
                }
                break;
            case 10:
                {   int ret = slice_from_s(z, 1, s_9); /* <-, line 63 */
                    if (ret < 0) return ret;
                }
                break;
            case 11:
                {   int ret = slice_from_s(z, 1, s_10); /* <-, line 64 */
                    if (ret < 0) return ret;
                }
                break;
            case 12:
                {   int ret = slice_from_s(z, 1, s_11); /* <-, line 65 */
                    if (ret < 0) return ret;
                }
                break;
            case 13:
                {   int ret = slice_from_s(z, 1, s_12); /* <-, line 66 */
                    if (ret < 0) return ret;
                }
                break;
            case 14:
                {   int ret = slice_from_s(z, 1, s_13); /* <-, line 67 */
                    if (ret < 0) return ret;
                }
                break;
            case 15:
                {   int ret = slice_from_s(z, 1, s_14); /* <-, line 68 */
                    if (ret < 0) return ret;
                }
                break;
            case 16:
                {   int ret = skip_utf8(z->p, z->c, 0, z->l, 1);
                    if (ret < 0) goto lab0;
                    z->c = ret; /* next, line 69 */
                }
                break;
        }
        continue;
    lab0:
        z->c = c1;
        break;
    }
    return 1;
}

static int r_exception(struct SN_env * z) {
    int among_var;
    z->bra = z->c; /* [, line 74 */
    among_var = find_among(z, a_1, 125); /* substring, line 74 */
    if (!(among_var)) return 0;
    z->ket = z->c; /* ], line 74 */
    if (z->c < z->l) return 0; /* atlimit, line 74 */
    switch(among_var) {
        case 0: return 0;
        case 1:
            {   int ret = slice_from_s(z, 3, s_15); /* <-, line 79 */
                if (ret < 0) return ret;
            }
            break;
        case 2:
            {   int ret = slice_from_s(z, 3, s_16); /* <-, line 83 */
                if (ret < 0) return ret;
            }
            break;
        case 3:
            {   int ret = slice_from_s(z, 3, s_17); /* <-, line 89 */
                if (ret < 0) return ret;
            }
            break;
        case 4:
            {   int ret = slice_from_s(z, 4, s_18); /* <-, line 93 */
                if (ret < 0) return ret;
            }
            break;
        case 5:
            {   int ret = slice_from_s(z, 5, s_19); /* <-, line 97 */
                if (ret < 0) return ret;
            }
            break;
        case 6:
            {   int ret = slice_from_s(z, 5, s_20); /* <-, line 101 */
                if (ret < 0) return ret;
            }
            break;
        case 7:
            {   int ret = slice_from_s(z, 5, s_21); /* <-, line 106 */
                if (ret < 0) return ret;
            }
            break;
        case 8:
            {   int ret = slice_from_s(z, 5, s_22); /* <-, line 111 */
                if (ret < 0) return ret;
            }
            break;
    }
    return 1;
}

static int r_R1(struct SN_env * z) {
    if (!(z->I[0] <= z->c)) return 0;
    return 1;
}

static int r_nouns_suffixes(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 119 */
        z->ket = z->c; /* [, line 120 */
        among_var = find_among_b(z, a_2, 170); /* substring, line 120 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 120 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 120 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 125 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 127 */
                break;
            case 2:
                {   int ret = slice_del(z); /* delete, line 131 */
                    if (ret < 0) return ret;
                }
                z->ket = z->c; /* [, line 132 */
                if (!(eq_s_b(z, 2, s_23))) goto lab0;
                z->bra = z->c; /* ], line 132 */
                {   int ret = slice_from_s(z, 2, s_24); /* <-, line 132 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 133 */
                break;
            case 3:
                {   int ret = slice_del(z); /* delete, line 155 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 156 */
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_nouns_endings(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 165 */
        z->ket = z->c; /* [, line 166 */
        among_var = find_among_b(z, a_3, 136); /* substring, line 166 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 166 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 166 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 169 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_25); /* <+, line 170 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 2:
                {   int ret = slice_del(z); /* delete, line 174 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_26); /* <+, line 175 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 3:
                {   int ret = slice_del(z); /* delete, line 179 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_27); /* <+, line 180 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 4:
                {   int ret = slice_del(z); /* delete, line 184 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_28); /* <+, line 185 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 5:
                {   int ret = slice_del(z); /* delete, line 189 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_29); /* <+, line 190 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 6:
                {   int ret = slice_del(z); /* delete, line 194 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 1, s_30); /* <+, line 195 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 7:
                {   int ret = slice_del(z); /* delete, line 199 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_31); /* <+, line 200 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 8:
                {   int ret = slice_del(z); /* delete, line 204 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_32); /* <+, line 205 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 9:
                {   int ret = slice_del(z); /* delete, line 209 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_33); /* <+, line 210 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 10:
                {   int ret = slice_del(z); /* delete, line 214 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_34); /* <+, line 215 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 11:
                {   int ret = slice_del(z); /* delete, line 219 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 1, s_35); /* <+, line 220 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 12:
                {   int ret = slice_del(z); /* delete, line 224 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_36); /* <+, line 225 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 13:
                {   int ret = slice_del(z); /* delete, line 229 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 1, s_37); /* <+, line 230 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 14:
                {   int ret = slice_del(z); /* delete, line 234 */
                    if (ret < 0) return ret;
                }
                {   int m_test = z->l - z->c; /* test, line 235 */
                    if (in_grouping_b_U(z, g_aeiou, 97, 117, 0)) goto lab0;
                    z->c = z->l - m_test;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_38); /* <+, line 235 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                {   int m_test = z->l - z->c; /* test, line 236 */
                    if (!(eq_s_b(z, 1, s_39))) goto lab0;
                    z->c = z->l - m_test;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_40); /* <+, line 236 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                {   int m_test = z->l - z->c; /* test, line 237 */
                    if (in_grouping_b_U(z, g_consonant, 98, 382, 0)) goto lab0;
                    z->c = z->l - m_test;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_41); /* <+, line 237 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 238 */
                break;
            case 15:
                {   int ret = slice_del(z); /* delete, line 242 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_42); /* <+, line 243 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 16:
                {   int ret = slice_del(z); /* delete, line 247 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_43); /* <+, line 248 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 17:
                {   int ret = slice_del(z); /* delete, line 258 */
                    if (ret < 0) return ret;
                }
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_adjectives_suffixes(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 265 */
        z->ket = z->c; /* [, line 266 */
        among_var = find_among_b(z, a_4, 77); /* substring, line 266 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 266 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 266 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 281 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 282 */
                break;
            case 2:
                {   int ret = slice_del(z); /* delete, line 286 */
                    if (ret < 0) return ret;
                }
                {   int ret = r_alternance_i(z);
                    if (ret == 0) goto lab0; /* call alternance_i, line 287 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 288 */
                break;
            case 3:
                {   int ret = slice_del(z); /* delete, line 292 */
                    if (ret < 0) return ret;
                }
                {   int m2 = z->l - z->c; (void)m2; /* or, line 293 */
                    z->ket = z->c; /* [, line 293 */
                    if (!(eq_s_b(z, 2, s_44))) goto lab2;
                    z->bra = z->c; /* ], line 293 */
                    {   int ret = slice_from_s(z, 1, s_45); /* <-, line 293 */
                        if (ret < 0) return ret;
                    }
                    goto lab1;
                lab2:
                    z->c = z->l - m2;
                    z->ket = z->c; /* [, line 293 */
                    if (!(eq_s_b(z, 2, s_46))) goto lab0;
                    z->bra = z->c; /* ], line 293 */
                    {   int ret = slice_from_s(z, 2, s_47); /* <-, line 293 */
                        if (ret < 0) return ret;
                    }
                }
            lab1:
                z->B[0] = 1; /* set found_suffix, line 294 */
                break;
            case 4:
                {   int ret = slice_del(z); /* delete, line 298 */
                    if (ret < 0) return ret;
                }
                {   int m3 = z->l - z->c; (void)m3; /* or, line 299 */
                    z->ket = z->c; /* [, line 299 */
                    if (!(eq_s_b(z, 2, s_48))) goto lab4;
                    z->bra = z->c; /* ], line 299 */
                    {   int ret = slice_from_s(z, 1, s_49); /* <-, line 299 */
                        if (ret < 0) return ret;
                    }
                    goto lab3;
                lab4:
                    z->c = z->l - m3;
                    z->ket = z->c; /* [, line 299 */
                    if (!(eq_s_b(z, 2, s_50))) goto lab5;
                    z->bra = z->c; /* ], line 299 */
                    {   int ret = slice_from_s(z, 2, s_51); /* <-, line 299 */
                        if (ret < 0) return ret;
                    }
                    goto lab3;
                lab5:
                    z->c = z->l - m3;
                    z->ket = z->c; /* [, line 299 */
                    if (!(eq_s_b(z, 2, s_52))) goto lab0;
                    z->bra = z->c; /* ], line 299 */
                    {   int ret = slice_from_s(z, 1, s_53); /* <-, line 299 */
                        if (ret < 0) return ret;
                    }
                }
            lab3:
                z->B[0] = 1; /* set found_suffix, line 300 */
                break;
            case 5:
                {   int ret = slice_del(z); /* delete, line 304 */
                    if (ret < 0) return ret;
                }
                z->ket = z->c; /* [, line 305 */
                if (!(eq_s_b(z, 2, s_54))) goto lab0;
                z->bra = z->c; /* ], line 305 */
                {   int ret = slice_from_s(z, 1, s_55); /* <-, line 305 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 306 */
                break;
            case 6:
                {   int ret = slice_del(z); /* delete, line 310 */
                    if (ret < 0) return ret;
                }
                {   int m4 = z->l - z->c; (void)m4; /* or, line 311 */
                    z->ket = z->c; /* [, line 311 */
                    if (!(eq_s_b(z, 2, s_56))) goto lab7;
                    z->bra = z->c; /* ], line 311 */
                    {   int ret = slice_from_s(z, 1, s_57); /* <-, line 311 */
                        if (ret < 0) return ret;
                    }
                    goto lab6;
                lab7:
                    z->c = z->l - m4;
                    z->ket = z->c; /* [, line 312 */
                    if (!(eq_s_b(z, 4, s_58))) goto lab8;
                    z->bra = z->c; /* ], line 312 */
                    {   int ret = slice_from_s(z, 2, s_59); /* <-, line 312 */
                        if (ret < 0) return ret;
                    }
                    goto lab6;
                lab8:
                    z->c = z->l - m4;
                    z->ket = z->c; /* [, line 313 */
                    if (!(eq_s_b(z, 4, s_60))) goto lab9;
                    z->bra = z->c; /* ], line 313 */
                    {   int ret = slice_from_s(z, 2, s_61); /* <-, line 313 */
                        if (ret < 0) return ret;
                    }
                    goto lab6;
                lab9:
                    z->c = z->l - m4;
                    z->ket = z->c; /* [, line 314 */
                    if (!(eq_s_b(z, 2, s_62))) goto lab10;
                    z->bra = z->c; /* ], line 314 */
                    {   int ret = slice_from_s(z, 1, s_63); /* <-, line 314 */
                        if (ret < 0) return ret;
                    }
                    goto lab6;
                lab10:
                    z->c = z->l - m4;
                    z->ket = z->c; /* [, line 315 */
                    if (!(eq_s_b(z, 2, s_64))) goto lab0;
                    z->bra = z->c; /* ], line 315 */
                    {   int ret = slice_from_s(z, 2, s_65); /* <-, line 315 */
                        if (ret < 0) return ret;
                    }
                }
            lab6:
                z->B[0] = 1; /* set found_suffix, line 316 */
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_adjectives_endings(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 323 */
        z->ket = z->c; /* [, line 324 */
        among_var = find_among_b(z, a_5, 111); /* substring, line 324 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 324 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 324 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 328 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_66); /* <+, line 329 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 2:
                {   int ret = slice_del(z); /* delete, line 334 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_67); /* <+, line 335 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 3:
                {   int ret = slice_del(z); /* delete, line 340 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_68); /* <+, line 341 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 4:
                {   int ret = slice_del(z); /* delete, line 347 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_69); /* <+, line 348 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 5:
                {   int ret = slice_del(z); /* delete, line 354 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_70); /* <+, line 355 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 6:
                {   int ret = slice_del(z); /* delete, line 361 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_71); /* <+, line 362 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 7:
                {   int ret = slice_del(z); /* delete, line 366 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_72); /* <+, line 367 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 8:
                {   int ret = slice_del(z); /* delete, line 373 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_73); /* <+, line 374 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 9:
                {   int ret = slice_del(z); /* delete, line 380 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_74); /* <+, line 381 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 10:
                {   int ret = slice_del(z); /* delete, line 386 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_75); /* <+, line 387 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 11:
                {   int ret = slice_del(z); /* delete, line 392 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_76); /* <+, line 393 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 12:
                {   int ret = slice_del(z); /* delete, line 397 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_77); /* <+, line 398 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_verbs_suffixes(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 405 */
        z->ket = z->c; /* [, line 406 */
        if (z->c <= z->lb || z->p[z->c - 1] != 116) goto lab0;
        among_var = find_among_b(z, a_6, 15); /* substring, line 406 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 406 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 406 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                z->B[0] = 1; /* set found_suffix, line 411 */
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_verbs_endings(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 418 */
        z->ket = z->c; /* [, line 419 */
        among_var = find_among_b(z, a_7, 345); /* substring, line 419 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 419 */
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 425 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_78); /* <+, line 426 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 2:
                {   int ret = slice_del(z); /* delete, line 432 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_79); /* <+, line 433 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 3:
                {   int ret = slice_del(z); /* delete, line 439 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 1, s_80); /* <+, line 440 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 4:
                {   int ret = slice_del(z); /* delete, line 447 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_81); /* <+, line 448 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 5:
                {   int ret = slice_del(z); /* delete, line 456 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_82); /* <+, line 457 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 6:
                {   int ret = slice_del(z); /* delete, line 463 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_83); /* <+, line 464 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 7:
                {   int ret = slice_del(z); /* delete, line 470 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_84); /* <+, line 471 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 8:
                {   int ret = slice_del(z); /* delete, line 475 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_85); /* <+, line 476 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 9:
                {   int ret = slice_del(z); /* delete, line 482 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 6, s_86); /* <+, line 483 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 10:
                {   int ret = slice_del(z); /* delete, line 488 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_87); /* <+, line 489 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 11:
                {   int ret = slice_del(z); /* delete, line 495 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 5, s_88); /* <+, line 496 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 12:
                {   int ret = slice_del(z); /* delete, line 501 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 2, s_89); /* <+, line 502 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 13:
                {   int ret = slice_del(z); /* delete, line 509 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_90); /* <+, line 510 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 14:
                {   int ret = slice_del(z); /* delete, line 517 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 4, s_91); /* <+, line 518 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 15:
                {   int ret = slice_del(z); /* delete, line 524 */
                    if (ret < 0) return ret;
                }
                {   int c_keep = z->c;
                    int ret = insert_s(z, z->c, z->c, 3, s_92); /* <+, line 525 */
                    z->c = c_keep;
                    if (ret < 0) return ret;
                }
                break;
            case 16:
                {   int ret = slice_del(z); /* delete, line 533 */
                    if (ret < 0) return ret;
                }
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_adverbs_suffixes(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1; /* do, line 540 */
        z->ket = z->c; /* [, line 541 */
        if (z->c - 1 <= z->lb || z->p[z->c - 1] >> 5 != 3 || !((37782048 >> (z->p[z->c - 1] & 0x1f)) & 1)) goto lab0;
        among_var = find_among_b(z, a_8, 6); /* substring, line 541 */
        if (!(among_var)) goto lab0;
        z->bra = z->c; /* ], line 541 */
        {   int ret = r_R1(z);
            if (ret == 0) goto lab0; /* call R1, line 541 */
            if (ret < 0) return ret;
        }
        switch(among_var) {
            case 0: goto lab0;
            case 1:
                {   int ret = slice_del(z); /* delete, line 544 */
                    if (ret < 0) return ret;
                }
                z->B[0] = 1; /* set found_suffix, line 545 */
                break;
        }
    lab0:
        z->c = z->l - m1;
    }
    return 1;
}

static int r_end_vowel(struct SN_env * z) {
    z->ket = z->c; /* [, line 552 */
    if (in_grouping_b_U(z, g_vowel, 97, 367, 0)) return 0;
    z->bra = z->c; /* ], line 552 */
    {   int ret = slice_from_s(z, 0, 0); /* <-, line 552 */
        if (ret < 0) return ret;
    }
    z->ket = z->c; /* [, line 553 */
    if (in_grouping_b_U(z, g_vowel, 97, 367, 0)) return 0;
    z->bra = z->c; /* ], line 553 */
    {   int ret = slice_from_s(z, 0, 0); /* <-, line 553 */
        if (ret < 0) return ret;
    }
    return 1;
}

static int r_alternance_i(struct SN_env * z) {
    {   int m1 = z->l - z->c; (void)m1; /* or, line 557 */
        z->ket = z->c; /* [, line 557 */
        if (!(eq_s_b(z, 2, s_93))) goto lab1;
        z->bra = z->c; /* ], line 557 */
        {   int ret = slice_from_s(z, 1, s_94); /* <-, line 557 */
            if (ret < 0) return ret;
        }
        goto lab0;
    lab1:
        z->c = z->l - m1;
        z->ket = z->c; /* [, line 557 */
        if (!(eq_s_b(z, 2, s_95))) return 0;
        z->bra = z->c; /* ], line 557 */
        {   int ret = slice_from_s(z, 2, s_96); /* <-, line 557 */
            if (ret < 0) return ret;
        }
    }
lab0:
    return 1;
}

static int r_end_double(struct SN_env * z) {
    int among_var;
    {   int m_test = z->l - z->c; /* test, line 561 */
        if (z->c - 1 <= z->lb || z->p[z->c - 1] >> 5 != 3 || !((1401044 >> (z->p[z->c - 1] & 0x1f)) & 1)) return 0;
        among_var = find_among_b(z, a_9, 9); /* substring, line 561 */
        if (!(among_var)) return 0;
        z->c = z->l - m_test;
    }
    switch(among_var) {
        case 0: return 0;
        case 1:
            z->ket = z->c; /* [, line 563 */
            {   int ret = skip_utf8(z->p, z->c, z->lb, 0, -1);
                if (ret < 0) return 0;
                z->c = ret; /* next, line 563 */
            }
            z->bra = z->c; /* ], line 563 */
            {   int ret = slice_del(z); /* delete, line 563 */
                if (ret < 0) return ret;
            }
            break;
    }
    return 1;
}

extern int czech_UTF_8_stem(struct SN_env * z) {
    {   int c1 = z->c; /* or, line 571 */
        {   int ret = r_exception(z);
            if (ret == 0) goto lab1; /* call exception, line 571 */
            if (ret < 0) return ret;
        }
        goto lab0;
    lab1:
        z->c = c1;
        {   int c2 = z->c; /* not, line 572 */
            {   int ret = skip_utf8(z->p, z->c, 0, z->l, + 4);
                if (ret < 0) goto lab3;
                z->c = ret; /* hop, line 572 */
            }
            goto lab2;
        lab3:
            z->c = c2;
        }
        goto lab0;
    lab2:
        z->c = c1;
        {   int c3 = z->c; /* do, line 573 */
            {   int ret = r_mark_regions(z);
                if (ret == 0) goto lab4; /* call mark_regions, line 573 */
                if (ret < 0) return ret;
            }
        lab4:
            z->c = c3;
        }
        z->B[0] = 0; /* unset found_suffix, line 574 */
        z->lb = z->c; z->c = z->l; /* backwards, line 575 */

        {   int m4 = z->l - z->c; (void)m4; /* do, line 576 */
            {   int ret = r_nouns_suffixes(z);
                if (ret == 0) goto lab5; /* call nouns_suffixes, line 576 */
                if (ret < 0) return ret;
            }
        lab5:
            z->c = z->l - m4;
        }
        {   int m5 = z->l - z->c; (void)m5; /* or, line 577 */
            if (!(z->B[0])) goto lab7; /* Boolean test found_suffix, line 577 */
            goto lab6;
        lab7:
            z->c = z->l - m5;
            {   int m6 = z->l - z->c; (void)m6; /* do, line 577 */
                {   int ret = r_nouns_endings(z);
                    if (ret == 0) goto lab8; /* call nouns_endings, line 577 */
                    if (ret < 0) return ret;
                }
            lab8:
                z->c = z->l - m6;
            }
        }
    lab6:
        {   int m7 = z->l - z->c; (void)m7; /* or, line 578 */
            if (!(z->B[0])) goto lab10; /* Boolean test found_suffix, line 578 */
            goto lab9;
        lab10:
            z->c = z->l - m7;
            {   int m8 = z->l - z->c; (void)m8; /* do, line 578 */
                {   int ret = r_nouns_suffixes(z);
                    if (ret == 0) goto lab11; /* call nouns_suffixes, line 578 */
                    if (ret < 0) return ret;
                }
            lab11:
                z->c = z->l - m8;
            }
        }
    lab9:
        {   int m9 = z->l - z->c; (void)m9; /* or, line 580 */
            if (!(z->B[0])) goto lab13; /* Boolean test found_suffix, line 580 */
            goto lab12;
        lab13:
            z->c = z->l - m9;
            {   int m10 = z->l - z->c; (void)m10; /* do, line 580 */
                {   int ret = r_adjectives_suffixes(z);
                    if (ret == 0) goto lab14; /* call adjectives_suffixes, line 580 */
                    if (ret < 0) return ret;
                }
            lab14:
                z->c = z->l - m10;
            }
        }
    lab12:
        {   int m11 = z->l - z->c; (void)m11; /* or, line 581 */
            if (!(z->B[0])) goto lab16; /* Boolean test found_suffix, line 581 */
            goto lab15;
        lab16:
            z->c = z->l - m11;
            {   int m12 = z->l - z->c; (void)m12; /* do, line 581 */
                {   int ret = r_adjectives_endings(z);
                    if (ret == 0) goto lab17; /* call adjectives_endings, line 581 */
                    if (ret < 0) return ret;
                }
            lab17:
                z->c = z->l - m12;
            }
        }
    lab15:
        {   int m13 = z->l - z->c; (void)m13; /* or, line 582 */
            if (!(z->B[0])) goto lab19; /* Boolean test found_suffix, line 582 */
            goto lab18;
        lab19:
            z->c = z->l - m13;
            {   int m14 = z->l - z->c; (void)m14; /* do, line 582 */
                {   int ret = r_adjectives_suffixes(z);
                    if (ret == 0) goto lab20; /* call adjectives_suffixes, line 582 */
                    if (ret < 0) return ret;
                }
            lab20:
                z->c = z->l - m14;
            }
        }
    lab18:
        {   int m15 = z->l - z->c; (void)m15; /* or, line 584 */
            if (!(z->B[0])) goto lab22; /* Boolean test found_suffix, line 584 */
            goto lab21;
        lab22:
            z->c = z->l - m15;
            {   int ret = r_adverbs_suffixes(z);
                if (ret == 0) return 0; /* call adverbs_suffixes, line 584 */
                if (ret < 0) return ret;
            }
        }
    lab21:
        {   int m16 = z->l - z->c; (void)m16; /* or, line 587 */
            if (!(z->B[0])) goto lab24; /* Boolean test found_suffix, line 587 */
            goto lab23;
        lab24:
            z->c = z->l - m16;
            {   int m17 = z->l - z->c; (void)m17; /* do, line 587 */
                {   int ret = r_verbs_endings(z);
                    if (ret == 0) goto lab25; /* call verbs_endings, line 587 */
                    if (ret < 0) return ret;
                }
            lab25:
                z->c = z->l - m17;
            }
        }
    lab23:
        {   int m18 = z->l - z->c; (void)m18; /* or, line 588 */
            if (!(z->B[0])) goto lab27; /* Boolean test found_suffix, line 588 */
            goto lab26;
        lab27:
            z->c = z->l - m18;
            {   int m19 = z->l - z->c; (void)m19; /* do, line 588 */
                {   int ret = r_verbs_suffixes(z);
                    if (ret == 0) goto lab28; /* call verbs_suffixes, line 588 */
                    if (ret < 0) return ret;
                }
            lab28:
                z->c = z->l - m19;
            }
        }
    lab26:
        {   int m20 = z->l - z->c; (void)m20; /* do, line 590 */
            {   int ret = r_end_double(z);
                if (ret == 0) goto lab29; /* call end_double, line 590 */
                if (ret < 0) return ret;
            }
        lab29:
            z->c = z->l - m20;
        }
        {   int m21 = z->l - z->c; (void)m21; /* do, line 591 */
            {   int ret = r_end_vowel(z);
                if (ret == 0) goto lab30; /* call end_vowel, line 591 */
                if (ret < 0) return ret;
            }
        lab30:
            z->c = z->l - m21;
        }
        {   int m22 = z->l - z->c; (void)m22; /* do, line 592 */
            {   int ret = r_end_double(z);
                if (ret == 0) goto lab31; /* call end_double, line 592 */
                if (ret < 0) return ret;
            }
        lab31:
            z->c = z->l - m22;
        }
        z->c = z->lb;
    }
lab0:
    {   int c23 = z->c; /* do, line 595 */
        {   int ret = r_postlude(z);
            if (ret == 0) goto lab32; /* call postlude, line 595 */
            if (ret < 0) return ret;
        }
    lab32:
        z->c = c23;
    }
    return 1;
}

extern struct SN_env * czech_UTF_8_create_env(void) { return SN_create_env(0, 1, 1); }

extern void czech_UTF_8_close_env(struct SN_env * z) { SN_close_env(z, 0); }

