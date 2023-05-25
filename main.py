import streamlit as st
import clifford as cf
import sympy
from math import sqrt
from galgebra.ga import Ga
from galgebra.printer import latex
# from IPython.display import Math
# from PIL import Image
import plotly.graph_objects as go


sympy.init_printing(latex_printer=latex, use_latex='mathjax')
st.set_page_config(layout="wide", page_title="Lipschitz group", page_icon="https://sun9-42.userapi.com/impg/7NiGocZ0v7sfgC64EwvUja0uHuXryd1o4b9nEw/p4xqKr0l6D4.jpg?size=1000x1000&quality=95&sign=1965efc8c859cba15b0c9d427e8f8bd6&type=album")

st.markdown("""
<style>
.css-erpbzb.edgvbvh3
{
    visibility:  hidden;
}
.css-cio0dv.egzxvld1
{
    visibility:  hidden;
}
.css-z3au9t.egzxvld2
{
    visibility:  hidden;
}
.css-10pw50.egzxvld1
{
    visibility:  hidden;
}
</style>
""", unsafe_allow_html=True)


st.title(":violet[Разложение элементов группы Липшица в произведение векторов в алгебрах Клиффорда]")
st.subheader("Это веб-приложение для работы с группой Липшица в алгебрах Клиффорда.")
st.markdown("Вот что можно делать с помощью этого веб-приложения:")
st.markdown("- Проверить принадлежность элемента группе Липшица,")
st.markdown("- Разложить элемент группы Липшица в произведение векторов,")
st.markdown("- Визуализировать геометрическую интерпретацию результатов разложения (в случае $C\!\ell_{2,0}$).")


st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)


st.sidebar.image(
            "https://sun9-42.userapi.com/impg/7NiGocZ0v7sfgC64EwvUja0uHuXryd1o4b9nEw/p4xqKr0l6D4.jpg?size=1000x1000&quality=95&sign=1965efc8c859cba15b0c9d427e8f8bd6&type=album",
            width=150
        )

st.sidebar.write("")
st.sidebar.write("")



## Блок определения алгебры Клиффорда (включая левую панель)
n = 3
st.sidebar.write('Задайте сигнатуру алгебры Клиффорда:')
p = st.sidebar.number_input('Количество генераторов, дающих в квадрате +e :violet[(p)]', 0, n, value=n)
q = st.sidebar.number_input('Количество генераторов, дающих в квадрате -e :violet[(q)]', 0, n-p, value=0)

layout, blades = cf.Cl(p,q) # задаем алгебру Клиффорда Cl(p,q)
locals().update(blades)

values = blades.values()
elements = []
for value in values:
    elements.append(value)

generators = []
for value in elements:
    if value(1) != 0:
        generators.append(value)

blades_dict = dict((i, list(blades.keys())[i]) for i in range(len(blades)))


# Задаём символы генераторов алгебры для вывода результатов в формате Latex
xyz_1 = (x) = sympy.symbols('1 ', real=True)
xyz_2 = (x, y) = sympy.symbols('1 2', real=True)
xyz_3 = (x, y, z) = sympy.symbols('1 2 3', real=True)
eta_ii = [1 for i in range(p)] + [-1 for i in range(q)]

xyz = (x, y, z) = sympy.symbols('1 2 3', real=True)
o3d = Ga('e_1 e_2 e_3', g=[1, -1, -1], coords=xyz)
e_1, e_2, e_3 = o3d.mv()

latex_generators = e_1, e_2, e_3

if p + q == 1:
    latex_generators = sympy.S('e_1')

if p + q == 2:
    o3d = Ga('e_1 e_2', g=eta_ii, coords=xyz_2)
    e_1, e_2 = o3d.mv()
    latex_generators = e_1, e_2

if p + q == 3:
    o3d = Ga('e_1 e_2 e_3', g=eta_ii, coords=xyz_3)
    e_1, e_2, e_3 = o3d.mv()
    latex_generators = e_1, e_2, e_3


st.header(":violet[Рассматриваемая алгебра Клиффорда]", anchor='section-algebra')
st.markdown("Изменить сигнатуру $$(p,q)$$ алгебры Клиффорда можно на боковой панели слева.")

# Вывод информации о рассматриваемой алгебре Клиффорда
if p + q != 0:
    if p + q == 1:
        st.markdown(f"Рассматриваем алгебру Клиффорда $$C\!\ell$$ размерности  $$n={p+q}$$ и сигнатуры $$(p,q)=({p},{q})$$ над полем $\mathbb{{R}}$. Генератор этой алгебры $e_1$ удовлетворяет соотношению")
    else:
        st.markdown(f"Рассматриваем алгебру Клиффорда $$C\!\ell$$ размерности  $$n={p+q}$$ и сигнатуры $$(p,q)=({p},{q})$$ над полем $\mathbb{{R}}$. Генераторы этой алгебры ${latex_generators}$ удовлетворяют соотношениям")

    st.latex(r"e_a e_b + e_b e_a = 2\eta_{ab}e,")
    st.markdown("где $\eta=(\eta_{ab})$ -- диагональная матрица:")

    if p + q == 1:
        st.latex(f"\eta=(1).")
    else:
        st.latex(f"\eta={latex(o3d.g)}.")

else:
    st.markdown("Мы не рассматриваем случай $$(p,q)=(0,0)$$.")




## Блок проверки принадлежности элемента группе Липшица

st.header(":violet[Проверка принадлежности элемента группе Липшица]", anchor='section-check')

st.markdown("В этом блоке можно проверить, принадлежит ли заданный элемент группе Липшица. \
    Напомним, что группа Липшица $$\Gamma^{\pm}$$ имеет следующее определение:")
st.latex("\Gamma^{\pm}:=\{T\in C\!\ell^{(0)}\cup C\!\ell^{(1)}:\quad \exists T^{-1},\quad T C\!\ell^{1}T^{-1}\subseteq C\!\ell^{1}\},")
st.markdown("то есть произвольный элемент $T$ группы Липшица принадлежит чётному $C\!\ell^{(0)}$ или нечётному $C\!\ell^{(1)}$ подпространству, \
    является обратимым и сохраняет подпространство $C\!\ell^{1}$ первого ранга при присоединённом действии $ad_{T}(U):=TUT^{-1}$ для $U\in C\!\ell$.")
st.markdown("В окошко ниже можно вписать элемент алгебры Клиффорда, который хочется проверить.")


def if_invertible_LG(element):
  '''
  Проверяет обратимость элемента группы Липшица

  :element: элемент алгебры Клиффорда
  :return: True/False, если принадлежит/не принадлежит
  '''
  norm_value = ~element * element
  if ((norm_value(0) == norm_value) & (norm_value != 0)):
    return True
  else:
    return False


def inverse_LG(element):
  '''
  Находит обратный элемент для элемента группы Липшица

  :element: элемент группы Липшица
  :return: обратный элемент
  '''
  norm_value = ~element * element
  return ~element / norm_value


def if_even(element):
  '''
  Проверяет чётность элемента

  :element: элемент алгебры Клиффорда
  :return: True/False, если является/не является чётным
  '''
  if element.even - element == 0:
    return True
  else:
    return False


def if_odd(element):
  '''
  Проверяет нечётность элемента

  :element: элемент алгебры Клиффорда
  :return: True/False, если является/не является нечётным
  '''
  if element.odd - element == 0:
    return True
  else:
    return False


def if_preserves_grade1(element):
  '''
  Проверяет, сохраняет ли элемент подпространство ранга 1 при присоединённом действии

  :element: элемент алгебры Клиффорда
  :return: True/False, если сохраняет/не сохраняет
  '''
  for generator in generators:
    ad = element * generator * ~element
    if ad(1) - ad != 0:
      return False
  return True


def if_in_LG(element):
  '''
  Проверяет, принадлежит ли элемент группе Липшица

  :element: элемент алгебры Клиффорда
  :return: True, если принадлежит; если не принадлежит, причина
  '''
  invertible, even, odd, preserves_grade1  = if_invertible_LG(element), if_even(element), if_odd(element), if_preserves_grade1(element)

  if invertible & (even | odd) & preserves_grade1:
    return True
  
  if invertible == False:
    return "Элемент НЕ принадлежит группе Липшица: значение функции нормы элемента $$\psi(T)=\widetilde{T}T$$ \
        не принадлежит подпространству ранга 0 или равно нулю."

  if (even | odd) == False:
    return "Элемент НЕ принадлежит группе Липшица: он не является ни чётным, ни нечётным."

  if preserves_grade1 == False:
    return "Элемент НЕ принадлежит группе Липшица: он не сохраняет подпространство ранга 1 при ad."


def factorization_n1(element):
  '''
  Раскладывает элемент группы Липшица в произведение векторов в случае n = 1

  :element: элемент группы Липшица
  :return: list() с множителями в разложении
  '''
  eta_11 = e1 ** 2

  if if_even(element) == True:
    return [(element(0) * eta_11 * e1), e1]
  else:
    return [element]


def factorization_n2(element):
  '''
  Раскладывает элемент группы Липшица в произведение векторов в случае n = 2

  :element: элемент группы Липшица
  :return: list() с множителями в разложении
  '''
  eta_11 = e1 ** 2

  u_dict = dict((i, element.value[i]) for i in range(len(element.value)))
  element_dict = {v: u_dict[k] for k, v in blades_dict.items()}

  if if_even(element) == True:
    return [e1, element_dict[''] * eta_11 * e1 + element_dict['e12'] * e2]
  else:
    return [element]


def factorization_n3(element):
  '''
  Раскладывает элемент группы Липшица в произведение векторов в случае n = 3

  :element: элемент группы Липшица
  :return: list() с множителями в разложении
  '''
  eta_11, eta_22, eta_33 = e1**2, e2**2, e3**2

  u_dict = dict((i, element.value[i]) for i in range(len(element.value)))
  element_dict = {v: u_dict[k] for k, v in blades_dict.items()}

  if (element(1) - element) == 0:
    return [element]

  elif if_even(element) == True:
    if element_dict['e12'] == 0:
      return [e3, -element_dict['e13'] * e1 - element_dict['e23'] * e2 + element_dict[''] * eta_33 * e3]

    elif element_dict['e13'] == 0:
      return [e2, -element_dict['e12'] * e1 + element_dict[''] * eta_22 * e2 + element_dict['e23'] * e3]

    elif element_dict['e23'] == 0:
      return [e1, element_dict[''] * eta_11 * e1 + element_dict['e12'] * e2 + element_dict['e13'] * e3]

    elif (element_dict['e13']) ** 2 * eta_11 + (element_dict['e23']) ** 2 * eta_22 != 0:
      lambda_coef = (element_dict['e13']) ** 2 * eta_11 + (element_dict['e23']) ** 2 * eta_22
      return (1 / lambda_coef) * (element_dict['e13'] * e1 + element_dict['e23'] * e2), \
        (element_dict[''] * element_dict['e13'] - element_dict['e12'] * element_dict['e23'] * eta_22) * e1 + \
            (element_dict[''] * element_dict['e23'] + element_dict['e12'] * element_dict['e13'] * eta_11) * e2 + lambda_coef * e3

    elif (element_dict['e12']) ** 2 * eta_11 + (element_dict['e23']) ** 2 * eta_33 != 0:
      lambda_coef = (element_dict['e12']) ** 2 * eta_11 + (element_dict['e23']) ** 2 * eta_33
      return (1 / lambda_coef) * (- element_dict['e12'] * e1 + element_dict['e23'] * e3), \
        (- element_dict[''] * element_dict['e12'] - element_dict['e13'] * element_dict['e23'] * eta_33) * e1 \
            - lambda_coef * e2 + (element_dict[''] * element_dict['e23'] - element_dict['e12'] * element_dict['e13'] * eta_11) * e3

    elif (element_dict['e12']) ** 2 * eta_22 + (element_dict['e13']) ** 2 * eta_33 != 0:
      lambda_coef = (element_dict['e12']) ** 2 * eta_22 + (element_dict['e13']) ** 2 * eta_33
      return (1 / lambda_coef) * (element_dict['e12'] * e2 + element_dict['e13'] * e3), \
        - lambda_coef * e1 + (element_dict[''] * element_dict['e12'] - element_dict['e13'] * element_dict['e23'] * eta_33) * e2 \
            + (element_dict[''] * element_dict['e13'] + element_dict['e12'] * element_dict['e23'] * eta_22) * e3

    else:
      return "Упс, что-то пошло не так... Но мы уже чиним это!"

  else:
    factors = [e1]
    factors.extend(factorization_n3(eta_11 * e1 * element))
    return factors


def factorization(element):
  '''
  Главная функция разложения элементов группы Липшица в произведение векторов

  :element: элемент алгебры Клиффорда
  :return: list() с множителями в разложении
  '''
  element = element + e1 - e1
  in_LG = if_in_LG(element)

  if in_LG == True:
    if p + q == 1:
      return factorization_n1(element)
    if p + q == 2:
      return factorization_n2(element)
    if p + q == 3:
      return factorization_n3(element)
  
  else:
    return in_LG


def beautiful_print_vector(vector):
  '''
  Выводит вектор в удобном формате для красивой печати

  :vector: вектор
  :return: вектор в формате latex
  '''
  return sum([x * y for x, y in zip(list(vector.value)[1:p+q+1], list(o3d.mv()))])


def find_max_min_coordinates(vectors):
  '''
  Находит максимальную и минимальную координату среди всех координат массива векторов

  :vector: list() с векторами
  :return: [минимальная координата, максимальная координата]
  '''
  all_coordinates = []
  for vector in vectors:
    all_coordinates.extend(list(dict((i, vector.value[i]) for i in range(len(vector.value))).values())[1:3])
  min_coordinate, max_coordinate = min(all_coordinates), max(all_coordinates)
  return [min_coordinate, max_coordinate]


def get_vector_coordinates(vector):
  '''
  Находит координаты вектора

  :vector: вектор (x1,x2)
  :return: [0, x1], [0, x2]
  '''
  end_coordinates = list(dict((i, vector.value[i]) for i in range(len(vector.value))).values())[1:3]
  return [[0, end_coordinates[0]], [0, end_coordinates[1]]]


def find_all_reflections(vectors, start):
  '''
  Находит все отражения для заданного элемента группы Липшица

  :vectors: list() с векторами
  :start: вектор, на который действуем
  :return: list() со всеми отражениями вектора start, включая start
  '''
  all_reflections = [start]
  applied_to = start
  vectors = list(reversed(vectors))

  for i in range(len(vectors)):
    applied_to = vectors[i].gradeInvol() * applied_to * vectors[i].inv()
    all_reflections.append(applied_to)

  return all_reflections


def figure_reflections(vectors, start):
  '''
  Строит анимированный график отражений

  :vectors: list() с векторами
  :start: вектор, на который действуем
  :return: анимированный график отражений
  '''
  all_reflections = find_all_reflections(vectors, start)
  all_coordinates = []
  for reflection in all_reflections:
    all_coordinates.append(get_vector_coordinates(reflection))
  
  fig_frames = []
  for i in range(len(all_coordinates)):
    fig_frames.append(go.Frame(data=[go.Scatter(x=all_coordinates[i][0], y=all_coordinates[i][1], marker= dict(size=10))]))
  
  if len(fig_frames) != 2:
    plot_title = "Композиция двух отражений"
  else:
    plot_title = "Отражение"
  
  data_to_show = [go.Scatter(x=all_coordinates[0][0], y=all_coordinates[0][1], marker= dict(size=10), line=dict(width=3, color="#ff6f69"), name="Отражаемый вектор")]

  vectors_coordinates = []
  hyper_coordinates = []
  for vector in vectors:
    vector_c = get_vector_coordinates(vector)
    vectors_coordinates.append(vector_c)
    if vector_c[0][1] == 0:
      hyper_coordinates.append([[-100, 100], [0, 0]])
    elif vector_c[1][1] == 0:
      hyper_coordinates.append([[0, 0], [-100, 100]])
    else:
      hyper_coordinates.append([[(-vector_c[1][1] * 100)/vector_c[0][1], (vector_c[1][1] * 100)/vector_c[0][1]], [100, -100]])

  colors = ['#32a852', '#3261a8']

  for i in range(len(vectors_coordinates)):
    data_to_show.append(go.Scatter(x=vectors_coordinates[i][0], y=vectors_coordinates[i][1], marker = dict(size=8), line=dict(color=colors[i], width=2), name=f"Вектор"))
    data_to_show.append(go.Scatter(x=hyper_coordinates[i][0], y=hyper_coordinates[i][1], line=dict(color=colors[i], width=2), marker = dict(symbol="asterisk"), name=f"Гиперплоскость"))

  fig = go.Figure(
    data=data_to_show,
    layout=go.Layout(
        xaxis=dict(range=[find_max_min_coordinates(all_reflections)[0] - 2, find_max_min_coordinates(all_reflections)[1] + 2]),
        yaxis=dict(range=[find_max_min_coordinates(all_reflections)[0] - 2, find_max_min_coordinates(all_reflections)[1] + 2]),
        autosize=False,
        width=650,
        height=550,
        title=plot_title,
        updatemenus=[dict(
            type="buttons",
            buttons=[{
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Пауза",
                "method": "animate"
            }, dict(label="Показать",
                          method="animate",
                          args=[None, {"frame": {"duration": 2500}, "showactive": False, "fromcurrent": True, "transition": {"duration": 400}}])
                     ])]
    ),
    frames=fig_frames)

  fig.update_yaxes(scaleanchor="x", scaleratio=1)
  fig.update_layout(title_x=0.5)
  fig.update_layout(template="plotly_dark")

  return fig


def figure_rotation(element, start):
  '''
  Строит график поворота

  :element: элемент группы Липшица
  :start: вектор, на который действуем
  :return: график поворота
  '''
  start_coordinates = get_vector_coordinates(start)

  end = element.gradeInvol() * start * element.inv()
  end_coordinates = get_vector_coordinates(end)

  fig = go.Figure(
    data=[go.Scatter(x=start_coordinates[0], y=start_coordinates[1], marker= dict(size=10), line=dict(width=3, color="#ff6f69"), name="Исходный вектор"), 
          go.Scatter(x=end_coordinates[0], y=end_coordinates[1], marker= dict(size=10), line=dict(width=3, color='#2ab7ca'), name="Результат поворота")], 
    layout=go.Layout(
        xaxis=dict(range=[find_max_min_coordinates([start, end])[0] - 2, find_max_min_coordinates([start, end])[1] + 2]),
        yaxis=dict(range=[find_max_min_coordinates([start, end])[0] - 2, find_max_min_coordinates([start, end])[1] + 2]),
        autosize=False,
        width=600,
        height=550,
        title="Поворот"))
  
  return fig



if p + q == 1:
    hint = "e1 или 5 + 2*e1"
    default_user_input = "e1"

if p + q == 2:
    hint = "e1 + e2" + " или " + "1 + 2\*e12" + " или " + "5 + 2\*e1"
    default_user_input = "1 + 2*e12"

if p + q == 3:
    hint = "e1 + e2 + e3 + e123" + " или " + "e1 + e2 + e3 + 2\*e123" + " или " + "1 + e123"
    default_user_input = "e1 + e2 + e3 + 2*e123"

if p + q != 0:
    st.caption(f"Подсказка: можно попробовать ввести элемент {hint}")
    user_input = st.text_input("Элемент алгебры Клиффорда", default_user_input, key='input_arb_el')
    st.write(f"Вы ввели элемент {user_input}.")
    try:
        input_element = eval(user_input)

        if if_in_LG(input_element) == True:
            st.markdown(f"**:violet[Результат проверки: ]** **Элемент принадлежит группе Липшица.**")
        else:
            st.markdown(f"**:violet[Результат проверки: ]** {if_in_LG(input_element)}")
    except NameError:
        st.write("Элемент введён некорректно. Проверьте выбранную сигнатуру слева и введённый элемент.")
    except TypeError:
        st.write("Элемент введён некорректно. Посмотрите примеры формата ввода выше.")
    except SyntaxError:
        st.write("Ошибка: введена пустая строка.")
else:
    st.markdown("Сначала выберите сигнатуру алгебры слева: мы не рассматриваем случай $$(p,q)=(0,0)$$.")




## Блок разложения элементов группы Липшица в произведение векторов 
st.header(":violet[Разложение элемента группы Липшица в произведение векторов]", anchor='section-factorization')
st.markdown("Известно, что любой элемент группы Липшица может быть представлен в виде произведения не более чем $n$ \
    обратимых элементов первого ранга (векторов).")
st.markdown("В этом блоке можно найти разложение заданного элемента группы Липшица в произведение обратимых векторов. \
    В окошко ниже можно вписать элемент группы Липшица, который хочется разложить в произведение.")

if p + q == 1:
    hint_fact = "e1" + " или " + "5" + " или " + "5 + 2\*e1"
    default_input_lg = "5*e1"

if p + q == 2:
    hint_fact = "1 + 2\*e12" + " или "+ "(1/2) + (sqrt(3)/2)\*e12" + " или " + "e1 + e2" + " или " + "5 + 2\*e1"
    default_input_lg = "1 + 2*e12"

if p + q == 3:
    hint_fact =   "e1 + e2 + e3 + 2\*e123" + " или " + "e1 + (1/2)*e2 + sqrt(3)*e3 + e123" + " или " + "1 + e123"
    default_input_lg = "e1 + e2 + e3 + 2*e123"

if p + q != 0:
    st.caption(f"Подсказка: можно попробовать ввести элемент {hint_fact}")
    input_lg = st.text_input("Элемент алгебры Клиффорда", default_input_lg, key='input_lg')
    st.write(f"Вы ввели элемент {input_lg}.")
    try: 
        input_element_lg = eval(input_lg)
        factorization_result = factorization(input_element_lg)
        check_result = 1
        if type(factorization_result) != str:
            st.write(f"**:violet[Множители в разложении]** ({len(factorization_result)} шт.):")
            for i in range(len(factorization_result)):
                st.latex(f"{i+1}: \quad\quad {beautiful_print_vector(factorization_result[i])}")
                check_result *= factorization_result[i]
            if check_result == input_element_lg:
                st.markdown(f"**Проверка результата**: при перемножении полученных {len(factorization_result)} шт. множителей \
                    получаем элемент {check_result}, что совпадает с поданным на вход элементом.")
                st.download_button('Скачать файл с разложением', str(factorization_result))
            else:
                st.write(f"Проверка результата: упс! При перемножении полученных {len(factorization_result)} шт. множителей \
                    получаем элемент {check_result}, что НЕ совпадает с поданным на вход элементом.")

            if (p == 2) & (q == 0):
                st.subheader(":violet[Визуализация результатов разложения]")

                st.markdown("Поворот $\widehat{T} v T^{-1}$ для $T\in\Gamma^{\pm}$ и $v\in C\!\ell^1$ может быть представлен в виде композиции отражений, пользуясь разложением элемента $T$ в произведение векторов.")
                st.markdown(f"На рисунках ниже можно посмотреть, как $$T=$${input_element_lg} осуществляет поворот заданного вектора и как этот поворот может быть представлен в виде композиции отражений с помощью полученного выше разложения. В окошке ниже можно ввести вектор, который будем поворачивать.")


                st.caption(f"Подсказка: можно попробовать ввести вектор e1-e2")
                user_input_geom = st.text_input("Вектор", "e1-e2", key='input_el_geom')

                try:
                    input_element_geom = eval(user_input_geom)
                    if input_element_geom(1) == input_element_geom:
                        st.write(f"Будем поворачивать вектор {user_input_geom}.")
                        col1, col2 = st.columns([1, 1])
                        col1.plotly_chart(figure_rotation(input_element_lg, input_element_geom), theme=None)
                        col2.plotly_chart(figure_reflections(factorization_result, input_element_geom), theme=None)
                    else:
                        st.write("Введён не вектор.")

                except NameError:
                    st.write("Элемент введён некорректно. Проверьте выбранную сигнатуру слева и введённый элемент.")
                except TypeError:
                    st.write("Элемент введён некорректно. Посмотрите примеры формата ввода выше.")
                except SyntaxError:
                    st.write("Ошибка: введена пустая строка.")
        else:
            st.write(factorization_result)
    except NameError:
        st.write("Элемент введён некорректно. Проверьте выбранную сигнатуру слева и введённый элемент. Посмотреть примеры формата ввода можно выше.")
    except TypeError:
        st.write("Элемент введён некорректно. Посмотрите примеры формата ввода выше.")
    except AttributeError:
        st.write("Элемент введён некорректно. Посмотрите примеры формата ввода выше.")
    except SyntaxError:
        st.write("Ошибка: введена пустая строка.")

else:
    st.markdown("Сначала выберите сигнатуру алгебры слева: мы не рассматриваем случай $$(p,q)=(0,0)$$.")