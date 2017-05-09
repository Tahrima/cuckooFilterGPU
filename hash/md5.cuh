/**
 * \file md5.h
 */
#ifndef _MD5_H
#define _MD5_H

/**
 * \brief          MD5 context structure
 */
typedef struct {
	unsigned long total[2]; /*!< number of bytes processed  */
	unsigned long state[4]; /*!< intermediate digest state  */
	unsigned char buffer[64]; /*!< data block being processed */
} md5_context;

/**
 * \brief          MD5 context setup
 *
 * \param ctx      context to be initialized
 */
__device__
void md5_starts(md5_context *ctx);

/**
 * \brief          MD5 process buffer
 *
 * \param ctx      MD5 context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
__device__
void md5_update(md5_context *ctx, unsigned char *input, int ilen);

/**
 * \brief          MD5 final digest
 *
 * \param ctx      MD5 context
 * \param output   MD5 checksum result
 */
__device__
void md5_finish(md5_context *ctx, unsigned char *output);

/**
 * \brief          Output = MD5( input buffer )
 *
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   MD5 checksum result
 */
__device__
void md5(unsigned char *input, int ilen, unsigned char *output);

#endif /* md5.h */
