
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

#define _DEBUG

#ifndef COMDEF_H
#define COMDEF_H

#define OP_ADD 1
#define OP_QUERY 2
#define OP_ACCURACY 3

#define HASHFUN_MD5 1
#define HASHFUN_SHA 2
#define HASHFUN_NORM 3

#define MAX_HASHES 8

#define HASHSIZE_MD5 16
#define HASHSIZE_SHA 20
#define HASHSIZE_NORMAL 16

#define ERR_INV_HASHES_NUMBER -1
#define ERR_INV_HASH_FUNCTION -2

#define BUFFER_TRANSFER_SIZE  33554432 //32M
//#define BUFFER_TRANSFER_SIZE 65536 //64K
#define BUFFER_DEVICE_SIZE 33554432 //32M
//#define BUFFER_DEVICE_SIZE  130536//32M


#endif
