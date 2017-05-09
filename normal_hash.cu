/*  ===========================================================================
*
*                       BloomGPU version 1
*          University of British Columbia, Vancouver Canada.
*
*                  (C) 2008 All Rights Reserved
*
*                              NOTICE
*
* Permission to use, copy, modify, and distribute this software and
* its documentation for any purpose and without fee is hereby granted
* provided that the above copyright notice appear in all copies and
* that both the copyright notice and this permission notice appear in
* supporting documentation.
*
* Neither the University of British Columbia nor the authors make any representations about the
* suitability of this software for any purpose. This software is provided
* ''as is'' without express or implied warranty.
*
* ===========================================================================*/

#include "normal_hash.cuh"

__device__
unsigned int Normal_SLMHash(unsigned char* str, unsigned int len) {
	unsigned int hash = 0;
	unsigned int i, temp;
	unsigned count = len;

	if (len >= 4) {
		for (i = 0; i + 4 <= len; str += 4, i += 4) {
			temp = ((*str << 24) | (*(str + 1) << NORMAL_HASH_SIZE) | (*(str + 2) << 8) | (*(str + 3)));
			hash ^= temp;
			count -= 4;
		}
	}

	switch (count) {
		case 0:
			return hash;
		case 1:
			hash ^= (unsigned int) (*str);
			return hash;
		case 2:
			hash ^= (unsigned int) ((*str << 8) | (*(str + 1)));
			return hash;
		case 3:
			hash ^= (unsigned int) ((*str << NORMAL_HASH_SIZE) | (*(str + 1) << 8) | (*(str + 2)));
			return hash;
		default:
		//printf("Invalid key size\n");
			return 0;
	}
}

__device__
unsigned int Normal_BKDRHash(unsigned char* str, unsigned int len) {
	unsigned int seed = 131; /* 31 131 1313 13131 131313 etc.. */
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < len; str++, i++) {
		hash = (hash * seed) + (*str);
	}

	return hash;
}

__device__
unsigned int Normal_SDBMHash(unsigned char* str, unsigned int len) {
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < len; str++, i++) {
		hash = (*str) + (hash << 6) + (hash << NORMAL_HASH_SIZE) - hash;
	}

	return hash;
}

__device__
unsigned int Normal_DJBHash(unsigned char* str, unsigned int len) {
	unsigned int hash = 5381;
	unsigned int i = 0;

	for (i = 0; i < len; str++, i++) {
		hash = ((hash << 5) + hash) + (*str);
	}

	return hash;
}

__device__
unsigned int Normal_APHash(unsigned char* str, unsigned int len) {
	unsigned int hash = 0xAAAAAAAA;
	unsigned int i = 0;

	for (i = 0; i < len; str++, i++) {
		hash ^= ((i & 1) == 0) ? ((hash << 7) ^ (*str) ^ (hash >> 3)) : (~((hash << 11) ^ (*str) ^ (hash >> 5)));
	}

	return hash;
}


