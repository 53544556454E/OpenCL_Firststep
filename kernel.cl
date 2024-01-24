uint GetRandomUint(uint *random_state);
float GetRandomUnitFloat(uint *random_state);

kernel void RandomTest(global float *dst, global uint *rnd, uint plane_size)
{
    size_t gid = get_global_id(0);
    uint random_state = rnd[gid];
	float f;

	f = GetRandomUnitFloat(&random_state);
	dst[gid] = f;

	f = GetRandomUnitFloat(&random_state);
	dst[plane_size + gid] = f;

	f = GetRandomUnitFloat(&random_state);
	dst[plane_size * 2 + gid] = f;

    rnd[gid] = random_state;
}

uint GetRandomUint(uint *random_state)
{
	uint x = *random_state;

	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	
	return *random_state = x;
}

float GetRandomUnitFloat(uint *random_state)
{
	uint x = GetRandomUint(random_state);

	return (float)(x & 0x7fff) / (0x7fff + 1.0f);
}
