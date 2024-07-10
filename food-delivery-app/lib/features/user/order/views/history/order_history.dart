import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:food_delivery_app/common/controllers/filter_bar_controller.dart';
import 'package:food_delivery_app/common/widgets/bars/filter_bar.dart';
import 'package:food_delivery_app/common/widgets/bars/menu_bar.dart';
import 'package:food_delivery_app/common/widgets/misc/main_wrapper.dart';
import 'package:food_delivery_app/common/widgets/bars/search_bar.dart';
import 'package:food_delivery_app/common/widgets/app_bar/sliver_app_bar.dart';
import 'package:food_delivery_app/features/user/order/views/history/widgets/order_history_list.dart';
import 'package:food_delivery_app/utils/constants/colors.dart';
import 'package:food_delivery_app/utils/constants/icon_strings.dart';
import 'package:food_delivery_app/utils/constants/image_strings.dart';
import 'package:food_delivery_app/utils/constants/sizes.dart';
import 'package:food_delivery_app/utils/device/device_utility.dart';
import 'package:food_delivery_app/utils/hardcode/hardcode.dart';
import 'package:get/get.dart';

class OrderHistoryView extends StatefulWidget {
  @override
  _OrderHistoryViewState createState() => _OrderHistoryViewState();
}

class _OrderHistoryViewState extends State<OrderHistoryView> {
  final FilterBarController _filterBarController = Get.put(FilterBarController("All"));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          CSliverAppBar(
            title: "Orders",
            noLeading: true,
          ),

          SliverToBoxAdapter(
            child: Column(
              children: [
                MainWrapper(child: CSearchBar()),
                SizedBox(height: TSize.spaceBetweenItemsVertical,),

                MainWrapper(
                  rightMargin: 0,
                  child: FilterBar(
                    filters: ["All", "Active", "Completed", "Cancelled", "5", "4", "3", "2", "1"],
                    exclude: ["All", "Active", "Completed", "Cancelled"],
                    suffixIconStr: TIcon.unearnedStar,
                    suffixIconStrClicked: TIcon.fillStar,
                  ),
                ),
                SizedBox(height: TSize.spaceBetweenItemsVertical,),

                MainWrapper(
                  child: Container(
                    height: 1000,
                    child: Obx(() => OrderHistoryList(
                      orders: THardCode.getOrderList(),
                      selectedFilter: _filterBarController.selectedFilter.value,
                    )
                    ))
                ),

              ],
            ),
          )
        ],
      ),
    );
  }

}
