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

#ifndef NORMAL_HASH_H
#define NORMAL_HASH_H

#define NORMAL_HASH_SIZE 16

__device__
unsigned int Normal_SLMHash(unsigned char* str, unsigned int len);

__device__
unsigned int Normal_BKDRHash(unsigned char* str, unsigned int len);

__device__
unsigned int Normal_SDBMHash(unsigned char* str, unsigned int len);

__device__
unsigned int Normal_DJBHash(unsigned char* str, unsigned int len);

__device__
unsigned int Normal_APHash(unsigned char* str, unsigned int len);

#endif
