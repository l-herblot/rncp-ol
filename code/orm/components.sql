create table if not exists public.maker_type
(
    id serial primary key,
    maker_type_name character varying(255) collate pg_catalog."default"
);
insert into public.maker_type (id, maker_type_name) values
(1, 'Us'),
(2, 'Restaurant'),
(3, 'Wholesaler'),
(4, 'Grower'),
(5, 'Manufacturer');
create table if not exists public.maker
(
    id serial primary key,
    maker_name character varying(255) collate pg_catalog."default",
    id_maker_type integer references  public.maker_type (id) on delete cascade,
    active boolean
);
insert into public.maker (id, maker_name, id_maker_type, active) values
(1, 'AFoodI', 1, true),
(2, 'Forget your diet', 2, true),
(3, 'Chop Shop Butcher & Grill', 3, true),
(4, 'Vegetables seller 1', 3, true),
(5, 'Home Grown Cow', 4, true),
(6, 'Super Seed Sisters', 4, true),
(7, 'Bake me I''m famous', 5, true);
create table if not exists public.components
(
    id serial primary key,
    id_maker integer references  public.maker (id) on delete cascade,
    components_name character varying(255) collate pg_catalog."default",
    active boolean
);
insert into public.components (id, id_maker, components_name, active) values
(1, 1, 'Pastrami sandwich', true),
(2, 2, 'Pastrami sandwich', true),
(3, 3, 'Pastrami', true),
(4, 5, 'Beef', true),
(5, 4, 'Salad', true),
(6, 4, 'Pickle', true),
(7, 6, 'Lettuce', true),
(8, 6, 'Pickle', true);
create table if not exists public.elements
(
    id serial primary key,
    id_component_parent integer references  public.components (id) on delete cascade,
    id_component_child integer references  public.components (id) on delete cascade,
    quantity character varying(255) collate pg_catalog."default",
    active boolean
);
insert into elements (id_component_parent, id_component_child, quantity, active) values
(1, 3, '100g', true),
(3, 4, '10kg', true),
(1, 5, '50g', true),
(1, 6, '20g', true),
(5, 7, '1kg', true),
(6, 8, '0.5kg', true),
(2, 3, '120g', true),
(2, 5, '60g', true),
(2, 6, '30g', true);

with recursive get_ingredients (id, id_parent, quantity, depth, is_cycle, path) as (
    select el.id_component_child, el.id_component_parent, el.quantity, 0, false, ARRAY[el.id_component_child]
    from public.elements el
    where el.id_component_parent = 1
    union
    select el2.id_component_child, el2.id_component_parent, el2.quantity, depth + 1, el.id = ANY(el.path), el.path || el.id
    from public.elements el2
    join get_ingredients el  on el2.id_component_parent = el.id
)
select * from get_ingredients i
join public.components c on i.id = c.id