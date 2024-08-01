import 'package:flutter/material.dart';
import 'package:food_delivery_app/common/widgets/misc/list_check_empty.dart';
import 'package:food_delivery_app/common/widgets/misc/main_wrapper.dart';
import 'package:food_delivery_app/common/widgets/bars/search_bar.dart';
import 'package:food_delivery_app/common/widgets/app_bar/sliver_app_bar.dart';
import 'package:food_delivery_app/features/user/food/controllers/category/food_category_controller.dart';
import 'package:food_delivery_app/common/widgets/cards/food_card_gr.dart';
import 'package:food_delivery_app/utils/constants/icon_strings.dart';
import 'package:food_delivery_app/utils/constants/image_strings.dart';
import 'package:food_delivery_app/utils/constants/sizes.dart';
import 'package:get/get.dart';

class FoodCategoryView extends StatelessWidget {
  final FoodCategoryController _controller = Get.put(FoodCategoryController());

  @override
  Widget build(BuildContext context) {
    String category = _controller.category.value;
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          CSliverAppBar(
            title: category,
            noLeading: (category == "Liked") ? true : false,
          ),
          SliverToBoxAdapter(
            child: Stack(
              children: [
                MainWrapper(
                  child: Column(
                    children: [
                      CSearchBar(),
                      SizedBox(height: TSize.spaceBetweenSections,),
                      ListCheckEmpty(
                        child: GridView.count(
                          crossAxisCount: 2,
                          crossAxisSpacing: 10,
                          mainAxisSpacing: 10,
                          childAspectRatio: 14 / 16,
                          shrinkWrap: true,
                          physics: NeverScrollableScrollPhysics(),
                          children: [
                            for(int i = 0; i < 4; i++)
                              FoodCard(
                                type: FoodCardType.grid,
                                name: 'Pizza',
                                image: TImage.hcBurger1,
                                stars: 4.5,
                                originalPrice: 10.0,
                                salePrice: 7.5,
                                onTap: () {},
                                heart: 'assets/icons/heart.svg',
                              ),
                          ],
                        ),
                      )
                    ],
                  ),
                )
              ],
            ),
          )
        ],
      ),
    );
  }
}
