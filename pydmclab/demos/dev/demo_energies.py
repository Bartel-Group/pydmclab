def get_gs_elemental_energies_from_mp(remake=False):
    fjson = os.path.join(DATA_DIR, 'mus_from_mp_no_corrections.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    mus = mus_at_0K()
    
    mp_pbe_mus = mus['mp']['pbe']
    
    mpq = MPQuery(api_key=API_KEY)
    
    mp_mus = {}
    for el in mp_pbe_mus:
        print(el)
        my_mu = mp_pbe_mus[el]
        el += '1'
        query = mpq.get_data_for_comp(el, 
                                      only_gs=True,
                                      dict_key='cmpd')
        
        mp_mu = query[el]['E_mp']
        mp_mus[el[:-1]] = mp_mu

    return write_json(mp_mus, fjson)
    
def compare_my_mus_to_mp_mus():
    """
    TO DO:
        - quick tests worked but should do something more significant
    """
    
    mus = mus_at_0K()
    
    my_mp_pbe_mus = mus['mp']['pbe']
    my_mp_pbe_mus = {el : my_mp_pbe_mus[el]['mu'] for el in my_mp_pbe_mus}
    
    mp_gs_mus_no_corrections = get_gs_elemental_energies_from_mp(remake=False)
        
    my_mus = []
    mp_mus = []
    for el in my_mp_pbe_mus:
        my_mu = my_mp_pbe_mus[el]
        mp_mu = mp_gs_mus_no_corrections[el]
        
        my_mus.append(my_mu)
        mp_mus.append(mp_mu)
        
        diff = abs(my_mu - mp_mu)
        
        if diff > 0.1:
            print(el)
            print('ME = %.2f' % my_mu)
            print('MP = %.2f' % mp_mu)
            print('\n')
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    xlim = min(my_mus + mp_mus), max(my_mus + mp_mus)
    ylim = xlim
    
    ax = plt.scatter(my_mus, mp_mus,
                     color='white',
                     edgecolor='blue')
    
    ax = plt.plot(xlim, xlim, color='black', lw=1, ls='--')
    
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)
    
    ax = plt.xlabel('my mus (eV/at)')
    ax = plt.ylabel('MP gs energies (eV/at)')
    
    
    fig.savefig(os.path.join(FIG_DIR, 'my_mus_vs_mp_mus.png'))

    
    return