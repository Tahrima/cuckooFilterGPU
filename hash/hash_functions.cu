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
#include <stdio.h>

#include "hash_functions.cuh"
#include "comdef.cuh"

#include "md5.cu"
#include "sha1.cu"
#include "normal_hash.cu"

__device__
void hash_Normal(unsigned char* inputBuffer,
                unsigned int bufferSize,
                unsigned int maxIndex,
		            unsigned int* result) {

	unsigned int firstHash = Normal_APHash(inputBuffer, bufferSize);
	*result = firstHash % maxIndex;
}

__device__
void hash_SHA(unsigned char* inputBuffer,
                      unsigned int bufferSize,
                      unsigned int maxIndex,
		                  unsigned int* result) {

	unsigned int shaHash[HASHSIZE_SHA];
	sha1(inputBuffer, bufferSize, (unsigned char*) shaHash);
  *result = shaHash[0] % maxIndex;
}

__device__
void hash_MD5(unsigned char* inputBuffer,
                      unsigned int bufferSize,
                      unsigned int maxIndex,
		                  unsigned int* result) {

	unsigned int md5Hash[HASHSIZE_MD5];
	md5(inputBuffer, bufferSize, (unsigned char*) md5Hash);
	*result = md5Hash[0] % maxIndex;
}

__device__
int hash_item(unsigned char* inputBuffer,
              unsigned int bufferSize,
              unsigned int maxIndex,
		          int hashFunction,
              unsigned int* result) {
	switch (hashFunction) {
	case HASHFUN_NORM:
		hash_Normal(inputBuffer, bufferSize, maxIndex, result);
		break;
	case HASHFUN_MD5:
		hash_MD5(inputBuffer, bufferSize, maxIndex, result);
		break;
	case HASHFUN_SHA:
		hash_SHA(inputBuffer, bufferSize, maxIndex, result);
		break;
	default:
		return ERR_INV_HASH_FUNCTION;
	}

	return 0;
}
