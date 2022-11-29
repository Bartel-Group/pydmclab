USE_INSTALLED_PYDMC = True
if USE_INSTALLED_PYDMC:
    from pydmc.StrucTools import StrucTools, SiteTools
    from pydmc.MPQuery import MPQuery    
else:
    from StrucTools import StrucTools, SiteTools
    from MPQuery import MPQuery

from pymatgen.core.structure import Structure

MPID = 'mp-19417'
API_KEY = '***REMOVED***'
    
def demo_basic_structure_manipulations(mpid=MPID):
    print('\n')

    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)
    
    print('\nformula = %s' % st.formula)
    print('compact formula = %s' % st.compact_formula)
    print('structure has %s' % st.els)
    print('downloaded structure has %i sites' % len(st.structure))
    st.make_supercell([1,2,3])
    print('supercell has %i sites' % len(st.structure))
    
def demo_ox_state_decoration(mpid=MPID):
    print('\n')
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)
    
    site = SiteTools(s, 0)
    print('from MP, site 0 is %s' % site.ion)
    
    s = st.decorate_with_ox_states
    
    print('after auto oxi state decoration, site 10 is %s' % SiteTools(s, 10).ion)
    
    st = StrucTools(s, ox_states={'Fe' : 2,
                                  'Ti' : 4,
                                  'O' : -2})
    
    s = st.decorate_with_ox_states
    
    print('after forcing Fe to be ox state = 2 (bad idea)!, site 10 is %s' % SiteTools(s, 10).ion)
    
    
    print(s[0].species)
    
def demo_replace_species_and_order(mpid=MPID):
    print('\n')
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)
    st.make_supercell([1,2,3])
    
    species_map = {'Fe' : {'Fe' : 0.875,
                           'Cr' : 0.125}}
    
    ordered_strucs = st.replace_species(species_map, n_strucs=10)
    
    print('generated %i structures' % len(ordered_strucs))
    
    for idx in ordered_strucs:
        s = Structure.from_dict(ordered_strucs[idx])
        print('\nstructure %i has formula %s' % (idx, StrucTools(s).formula))
        site = SiteTools(s, 12)
        print('site 13 has ion = %s, coords = %s with ox state = %s' % (site.ion, site.coords, site.ox_state))

def demo_dilute_vacancy(mpid=MPID):
    """
    Broken right now...
    """
    print('\n')
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)
        
    out = st.get_structures_with_dilute_vacancy(el_to_replace='O',
                                                n_strucs=10,
                                                structure=None)
    
    struc = Structure.from_dict(out[0])
    
    print(struc)
    

        
def main():
    demo_basic_structure_manipulations()
    demo_ox_state_decoration(mpid=MPID)
    demo_replace_species_and_order(mpid=MPID)
    demo_dilute_vacancy(mpid=MPID)
    return

if __name__ == '__main__':
    main()