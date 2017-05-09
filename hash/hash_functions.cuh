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

#ifndef HASHFUNCTIONS_H
#define HASHFUNCTIONS_H

//TODO Move *.h to .c and put documentation.

__device__
int hash_item(unsigned char* inputBuffer, unsigned int bufferSize, unsigned int maxIndex, int hashFunction, unsigned int* result);

#endif
